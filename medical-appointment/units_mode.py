"""Utterance-unit answer space for the medical pipeline.

Gold evidence spans are whole utterances: they start at the speech onset after
a silence and end at the last word.  So instead of asking the LLM for a free
quote and locating it, we cut the transcript into silence-delimited units,
show them numbered, and ask for unit ids (one unit, or up to three consecutive
ones).  Oracle on the training set: best single unit 0.69 tIoU, best 1-3
consecutive units 0.81 (whisper segments: 0.62 / 0.66).

Selection can be sharpened by a learned cross-encoder (reranker.py) that
scores (question, unit) pairs; its score is combined with the LLM's choice.
"""
import logging
import re
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)
Span = Tuple[float, float]

UNIT_GAP = 0.2          # split units where the gap between words is >= this (s)
MAX_UNITS_PER_SPAN = 3


def build_units(segments: Sequence[Dict], gap: float = UNIT_GAP) -> List[Dict]:
    words = [w for s in segments for w in s.get("words", [])]
    silences = segments[0].get("_silences", []) if segments else []
    onsets = [b for _, b in silences]
    groups, cur = [], []
    for w in words:
        if cur and (w["start"] - cur[-1]["end"]) >= gap:
            groups.append(cur)
            cur = []
        cur.append(w)
    if cur:
        groups.append(cur)
    units = []
    for g in groups:
        s, e = g[0]["start"], g[-1]["end"]
        if onsets:
            ns = min(onsets, key=lambda x: abs(x - s))
            if abs(ns - s) <= 0.5:
                s = ns
        if e <= s:
            e = s + 0.3
        units.append({"start": round(s, 2), "end": round(e, 2), "text": " ".join(w["word"] for w in g), "words": g})
    return units


SYSTEM_UNITS = """You are a meticulous clinical documentation auditor. You will read a transcript of a recorded consultation between a clinician and a patient, cut into short numbered utterances, then answer yes/no questions about what was actually said.

Rules:
- Answer ONLY from the transcript. Do not use outside medical knowledge.
- Answer "yes" only if the transcript explicitly establishes the statement. Many questions are near-misses: same drug but a different dose, different duration, different body side, different frequency, a plausible detail never actually agreed. If ANY specific detail differs, answer "no". If the topic never comes up, answer "no".
- Some questions are tag questions ("..., didn't it?") or statements; phrasing never implies the answer.
- For every "yes", give the utterance number(s) that state or confirm the fact: the single most literal utterance, or a range of at most three consecutive utterances when the fact is established across a question-and-answer exchange (e.g. "12-13"). Prefer the utterance that contains the specific detail asked about. Different questions may cite the same utterance.

Output format, one line per question, in order, nothing else:
1|yes|12
2|no|
3|yes|7-8"""


def build_prompt_units(units: Sequence[Dict], questions: Sequence[str]) -> str:
    lines = [f"[{i}] {u['text']}" for i, u in enumerate(units)]
    qs = [f"{i + 1}. {q}" for i, q in enumerate(questions)]
    return "TRANSCRIPT:\n" + "\n".join(lines) + "\n\nQUESTIONS:\n" + "\n".join(qs) + "\n\nAnswer every question in order."


_ROW = re.compile(r"^\s*(\d+)\s*\|\s*(yes|no|y|n)\s*\|?\s*(\d+)?\s*(?:-\s*(\d+))?\s*$", re.IGNORECASE)


def parse_units(text: str, n: int, n_units: int) -> List[Tuple[Optional[bool], Optional[Tuple[int, int]]]]:
    out: List[Tuple[Optional[bool], Optional[Tuple[int, int]]]] = [(None, None) for _ in range(n)]
    for line in text.splitlines():
        m = _ROW.match(line.strip())
        if not m:
            continue
        q = int(m.group(1)) - 1
        if not 0 <= q < n:
            continue
        ans = m.group(2).lower().startswith("y")
        rng = None
        if m.group(3) is not None:
            a = int(m.group(3))
            b = int(m.group(4)) if m.group(4) is not None else a
            a, b = min(a, b), max(a, b)
            b = min(b, a + MAX_UNITS_PER_SPAN - 1)
            if 0 <= a < n_units:
                rng = (a, min(b, n_units - 1))
        out[q] = (ans, rng)
    return out


RERANK_UNITS = """You are checking which utterance of a consultation transcript is the evidence for a yes/no question that was answered YES.
For each question you get a few candidate utterances. Pick the ONE utterance whose words most literally and directly state the fact the question asks about (same drug, same number, same finding, same action). Prefer the utterance containing the specific detail over a vague or paraphrased one.
Output one line per question, in order: <question number>|<utterance number>. Nothing else."""


def rerank_units(units: Sequence[Dict], questions: Sequence[str], parsed, llm_generate, raw_cache,
                 extra_candidates: Dict[int, List[int]]) -> Dict[int, int]:
    """Second LLM pass over candidate units.  Returns {question index: unit index}."""
    out: Dict[int, int] = {}
    try:
        items = []
        for qi, (ans, rng) in enumerate(parsed):
            if not ans:
                continue
            cands = set()
            if rng is not None:
                cands.update(range(rng[0], rng[1] + 1))
                cands.update(i for i in (rng[0] - 1, rng[1] + 1) if 0 <= i < len(units))
            cands.update(extra_candidates.get(qi, []))
            uniq = sorted(i for i in cands if 0 <= i < len(units))
            if uniq:
                items.append((qi, uniq))
        if not items:
            return out
        parts = []
        for qi, uniq in items:
            parts.append(f"Question {qi + 1}: {questions[qi]}\nCandidates:\n" + "\n".join(f"  [{i}] {units[i]['text']}" for i in uniq))
        user = "\n\n".join(parts) + "\n\nAnswer with <question number>|<utterance number> per question."
        text = llm_generate(RERANK_UNITS, user, 12 * len(items) + 20, raw_cache)
        allowed = dict(items)
        for line in text.splitlines():
            m = re.match(r"\s*(\d+)\s*\|\s*\[?(\d+)\]?", line)
            if m:
                qi, ui = int(m.group(1)) - 1, int(m.group(2))
                if qi in allowed and ui in allowed[qi]:
                    out[qi] = ui
    except Exception:
        logger.exception("unit re-rank failed")
    return out


def answer_all_units(segments: Sequence[Dict], questions: Sequence[str], deadline, raw_cache, cfg: Dict):
    """Drop-in for pipeline.answer_all when SPAN_STRATEGY['mode'] == 'units'."""
    import pipeline  # late import: pipeline imports this module

    n = len(questions)
    units = build_units(segments, float(cfg.get("unit_gap", UNIT_GAP)))
    if not units:
        return [False] * n, [None] * n
    parsed: List[Tuple[Optional[bool], Optional[Tuple[int, int]]]] = [(None, None) for _ in range(n)]
    if deadline is None or time.perf_counter() < deadline - pipeline.MAIN_RESERVE:
        try:
            text = pipeline._llm_generate(SYSTEM_UNITS, build_prompt_units(units, questions), pipeline.MAX_NEW_TOKENS, raw_cache)
            pipeline.LAST_RAW["text"] = text
            parsed = parse_units(text, n, len(units))
        except Exception:
            logger.exception("units LLM pass failed")
    else:
        logger.warning("no time for the LLM; lexical answers")

    # candidate units from the learned re-ranker / lexical / embedding, for the re-rank pass
    ce_scores: Dict[int, np.ndarray] = {}
    extra: Dict[int, List[int]] = {}
    use_ce = float(cfg.get("ce_weight", 0.0)) > 0
    if use_ce:
        try:
            import reranker
            for qi, (ans, rng) in enumerate(parsed):
                if ans:
                    sc = reranker.score(questions[qi], [u["text"] for u in units])
                    ce_scores[qi] = sc
                    extra[qi] = [int(i) for i in np.argsort(-sc)[:3]]
        except Exception:
            logger.exception("cross-encoder scoring failed")
            use_ce = False
    for qi, (ans, rng) in enumerate(parsed):
        if ans:
            lex = pipeline.lexical_span(units, questions[qi])
            if lex is not None:
                extra.setdefault(qi, []).extend(i for i, u in enumerate(units) if u["end"] > lex[0] and u["start"] < lex[1])

    chosen: Dict[int, int] = {}
    if cfg.get("rerank") and (deadline is None or time.perf_counter() < deadline - pipeline.RERANK_RESERVE):
        chosen = rerank_units(units, questions, parsed, pipeline._llm_generate, raw_cache, extra)

    answers: List[bool] = []
    spans: List[Optional[Span]] = []
    silences = segments[0].get("_silences", [])
    for qi, (ans, rng) in enumerate(parsed):
        if ans is None:
            ans, ids = pipeline.lexical_fallback(units, questions[qi])
            rng = (ids[0], ids[0]) if ids else None
        span = None
        if ans:
            a, b = rng if rng is not None else (None, None)
            if use_ce and qi in ce_scores:
                # combine: cross-encoder score + bonus for what the LLM (and the re-rank) chose
                s = ce_scores[qi].astype(np.float64).copy()
                s = (s - s.mean()) / (s.std() + 1e-6)
                if rng is not None:
                    s[rng[0]:rng[1] + 1] += float(cfg.get("llm_bonus", 1.5))
                if qi in chosen:
                    s[chosen[qi]] += float(cfg.get("rerank_bonus", 1.0))
                best = int(np.argmax(s))
                if rng is not None and rng[0] <= best <= rng[1]:
                    a, b = rng                       # keep the LLM's range when it contains the best unit
                else:
                    a, b = best, best
            elif qi in chosen:
                c = chosen[qi]
                if rng is None or not (rng[0] <= c <= rng[1]):
                    a, b = c, c
            if a is None:
                lex = pipeline.lexical_span(units, questions[qi])
                span = lex
            else:
                span = (units[a]["start"], units[b]["end"])
                if cfg.get("snap", 1) and silences:
                    span = pipeline.snap_span(span, silences, float(cfg.get("snap_start_tol", 0.4)), float(cfg.get("snap_end_tol", 0.2)))
        answers.append(bool(ans))
        spans.append(span)
    return answers, spans
