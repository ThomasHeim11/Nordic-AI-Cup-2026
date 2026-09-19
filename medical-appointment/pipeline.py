"""Medical appointment: local ASR + local LLM question answering with evidence.

Pipeline for one request:

1. ``transcribe`` -- decode the MP3 with PyAV (no ffmpeg needed) and run
   whisper-large-v3-turbo through MLX with segment timestamps.  Segments are
   the unit of evidence: gold spans are 1.3-5.5 s long (median 2.9 s), which is
   one or two whisper segments.
2. ``answer_all`` -- ONE chat completion answers all ten questions against a
   numbered transcript and cites the segment numbers it read each answer from.
   One prefill instead of ten is what keeps the request inside the 60 s budget.
3. ``span_from_segments`` -- the cited segments become the evidence interval.

Everything degrades instead of raising: no transcript -> guess "no" for
everything (balanced sets make that worth 0.5 on the accuracy half); no LLM ->
a lexical fallback that at least finds spans for questions that overlap the
transcript strongly.
"""

import json
import math
import logging
import os
import re
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

BACKEND = os.environ.get("MEDICAL_BACKEND", "mlx")     # "mlx" on the Mac, "torch" on CUDA nodes
ASR_REPO = os.environ.get("ASR_REPO", "mlx-community/whisper-large-v3-turbo")
LLM_REPO = os.environ.get("LLM_REPO", "mlx-community/Qwen2.5-7B-Instruct-4bit")
MAX_SPAN_SECONDS = 16.0
MAX_NEW_TOKENS = 420
RERANK_RESERVE = 14.0     # seconds that must remain before starting the re-rank pass
MAIN_RESERVE = 16.0       # seconds that must remain before starting the main LLM pass

Span = Tuple[float, float]


# --------------------------------------------------------------------------- #
# ASR
# --------------------------------------------------------------------------- #

def decode_mp3(audio_bytes: bytes, sampling_rate: int = 16000) -> np.ndarray:
    import io
    from faster_whisper.audio import decode_audio
    return decode_audio(io.BytesIO(audio_bytes), sampling_rate=sampling_rate)


def silence_intervals(audio: np.ndarray, sr: int = 16000, frame_s: float = 0.01,
                      min_len_s: float = 0.12) -> List[Tuple[float, float]]:
    """(start, end) of silences >= min_len_s, from a simple RMS energy gate.

    The conversations are synthesized, so utterance boundaries are clean gaps;
    gold evidence spans start at the speech onset after such a gap (within
    ~0.05 s) and end at the last word's end.
    """
    n = int(frame_s * sr)
    if len(audio) < 2 * n:
        return []
    m = len(audio) // n
    env = np.sqrt(np.mean(audio[: m * n].reshape(m, n) ** 2, axis=1))
    thr = float(np.percentile(env, 20)) * 1.5 + 1e-6
    sil = env < thr
    out, start = [], None
    for i, v in enumerate(sil):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if (i - start) * frame_s >= min_len_s:
                out.append((start * frame_s, i * frame_s))
            start = None
    if start is not None and (m - start) * frame_s >= min_len_s:
        out.append((start * frame_s, m * frame_s))
    return out


def snap_span(span: Span, silences: Sequence[Tuple[float, float]], start_tol: float = 0.6,
              end_tol: float = 0.3) -> Span:
    """Snap a span's start to the nearest speech onset and its end to the nearest speech offset."""
    if not silences:
        return span
    s, e = span
    onsets = [b for _, b in silences]          # speech starts where a silence ends
    offsets = [a for a, _ in silences]         # speech ends where a silence starts
    ns = min(onsets, key=lambda x: abs(x - s))
    if abs(ns - s) <= start_tol:
        s = ns
    ne = min(offsets, key=lambda x: abs(x - e))
    if abs(ne - e) <= end_tol:
        e = ne
    if e <= s:
        return span
    return (round(s, 2), round(e, 2))


def transcribe(audio_bytes: bytes) -> List[Dict]:
    """Return [{"start", "end", "text"}] segments with second timestamps.

    The segment list also carries the silence structure of the audio under
    the key "_silences" of the first segment (so callers that only pass the
    segment list around still have it).
    """
    audio = decode_mp3(audio_bytes)
    silences = silence_intervals(audio)
    if BACKEND == "torch":
        import backend_torch
        result = backend_torch.transcribe_words(audio)
    else:
        import mlx_whisper
        result = mlx_whisper.transcribe(
            audio, path_or_hf_repo=ASR_REPO, language="en", word_timestamps=True,
            condition_on_previous_text=False, fp16=True,
        )
    segments = []
    for s in result.get("segments", []):
        text = s.get("text", "").strip()
        if not text:
            continue
        words = [{"word": w["word"].strip(), "start": float(w["start"]), "end": float(w["end"])}
                 for w in s.get("words", []) if w.get("word", "").strip()]
        segments.append({"start": float(s["start"]), "end": float(s["end"]), "text": text, "words": words})
    if segments:
        segments[0]["_silences"] = silences
    return segments


# --------------------------------------------------------------------------- #
# LLM
# --------------------------------------------------------------------------- #

_LLM = {"model": None, "tok": None}
LAST_RAW = {"text": ""}   # raw reply of the last answer_all call, for debugging


def load_llm():
    if _LLM["model"] is None:
        from mlx_lm import load
        t0 = time.perf_counter()
        _LLM["model"], _LLM["tok"] = load(LLM_REPO)
        logger.info("loaded %s in %.1fs", LLM_REPO, time.perf_counter() - t0)
    return _LLM["model"], _LLM["tok"]


SYSTEM_PROMPT = """You are a meticulous clinical documentation auditor. You will read a numbered transcript of a recorded consultation between a clinician and a patient, then answer yes/no questions about what was actually said.

Rules:
- Answer ONLY from the transcript. Do not use outside medical knowledge to infer things that were not said.
- Answer "yes" only if the transcript explicitly establishes the statement in the question.
- Many questions are near-misses: the same drug at a different dose, a different duration, a different body side, a different frequency, a different lab value, or a plausible detail that was never actually agreed. If ANY specific detail in the question (number, unit, drug, timing, location, who does what) differs from the transcript, answer "no".
- If the topic is never mentioned at all, answer "no".
- Some questions are phrased as tag questions ("..., didn't it?") or statements; the phrasing never implies the answer.
- For every "yes", copy the exact words from the transcript that establish it, character-for-character from the numbered lines, plus the number of the line the quote starts on. Quote the passage where the fact is actually stated or confirmed. When the fact is established through an exchange (one speaker asks or proposes, the other confirms), quote both turns together, e.g. "So, this is your annual follow-up. It is, for the asthma." Keep it tight: only the words that carry the fact (typically 5-25 words), never a whole paragraph. Different questions may cite the same passage; do not avoid a passage because you already used it. For "no", give no quote.

Output format, one line per question, in order, nothing else:
1|yes|12|the exact quoted words from line 12
2|no||
3|yes|4|another verbatim quote"""


def build_prompt(segments: Sequence[Dict], questions: Sequence[str]) -> str:
    lines = [f"[{i}] ({s['start']:.1f}s-{s['end']:.1f}s) {s['text']}" for i, s in enumerate(segments)]
    qs = [f"{i + 1}. {q}" for i, q in enumerate(questions)]
    return "TRANSCRIPT:\n" + "\n".join(lines) + "\n\nQUESTIONS:\n" + "\n".join(qs) + "\n\nAnswer every question in order."


_ROW_RE = re.compile(r"^\s*(\d+)\s*\|\s*(yes|no|y|n)\s*\|\s*(\d*)\s*\|?\s*(.*)$", re.IGNORECASE)


def parse_answers(text: str, n: int) -> List[Tuple[Optional[bool], Optional[int], str]]:
    """Parse the compact reply into [(answer or None, line id or None, quote)]."""
    out: List[Tuple[Optional[bool], Optional[int], str]] = [(None, None, "") for _ in range(n)]
    for line in text.splitlines():
        m = _ROW_RE.match(line)
        if not m:
            continue
        q = int(m.group(1)) - 1
        if not 0 <= q < n:
            continue
        answer = m.group(2).lower().startswith("y")
        seg = int(m.group(3)) if m.group(3).isdigit() else None
        out[q] = (answer, seg, m.group(4).strip())
    return out


SPAN_STRATEGY = {"mode": "quote", "pad": 0.0, "rerank": 1, "seg_sub_min": 0.5, "rerank_quote": 0, "snap": 1, "snap_start_tol": 0.4, "snap_end_tol": 0.2,
                 # cross-encoder second opinion (see _ce_ensemble): held-out tIoU 0.526 -> 0.556 on question_train.csv
                 "ce_ens": 1, "ce_top": 2, "ce_margin": 1.0, "ce_grow": 0.5}

RERANK_SYSTEM = """You are checking which transcript line is the evidence for a yes/no question that was answered YES.
For each question you get a few candidate lines from the transcript. Pick the ONE line whose words most literally and directly state the fact the question asks about (same drug, same number, same finding, same action). Prefer the line containing the specific detail over a vague or paraphrased one.
Output one line per question, in order: <question number>|<line number>|<the exact words on that line that state the fact, copied verbatim, 3-20 words>. Nothing else."""


def _llm_generate(system: str, user: str, max_tokens: int, raw_cache: Optional[Dict[str, str]] = None) -> str:
    import hashlib
    key = hashlib.sha1((system + "\n" + user + (os.environ.get("LLM_HF_REPO", "Qwen/Qwen2.5-32B-Instruct") if BACKEND == "torch" else LLM_REPO)).encode()).hexdigest()
    if raw_cache is not None and key in raw_cache:
        return raw_cache[key]
    if BACKEND == "torch":
        import backend_torch
        text = backend_torch.chat(system, user, max_tokens)
    else:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        model, tok = load_llm()
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True)
        text = generate(model, tok, prompt=prompt, max_tokens=max_tokens, verbose=False, sampler=make_sampler(temp=0.0))
    if raw_cache is not None:
        raw_cache[key] = text
    return text


def rerank_evidence(segments: Sequence[Dict], questions: Sequence[str],
                    parsed: List[Tuple[Optional[bool], Optional[int], str]],
                    raw_cache: Optional[Dict[str, str]] = None) -> Dict[int, int]:
    """Second pass: for each yes, let the LLM choose among candidate lines.

    Returns {question index: (chosen segment index, verbatim words or "")}.  Never raises.
    """
    out: Dict[int, Tuple[int, str]] = {}
    try:
        items = []
        for qi, (ans, seg, quote) in enumerate(parsed):
            if not ans:
                continue
            cands = []
            if seg is not None and 0 <= seg < len(segments):
                cands.append(seg)
            qspan = locate_quote(segments, quote, seg)
            if qspan is not None:
                cands += [i for i, sg in enumerate(segments) if sg["end"] > qspan[0] and sg["start"] < qspan[1]]
            for i, _ in embed_rank(segments, questions[qi])[:2]:
                cands.append(i)
            lex = lexical_span(segments, questions[qi])
            if lex is not None:
                cands += [i for i, sg in enumerate(segments) if sg["end"] > lex[0] and sg["start"] < lex[1]]
            uniq = sorted(set(cands))
            if not uniq:
                continue
            items.append((qi, uniq))
        if not items:
            return out
        parts = []
        for qi, uniq in items:
            lines = "\n".join(f"  [{i}] {segments[i]['text']}" for i in uniq)
            parts.append(f"Question {qi + 1}: {questions[qi]}\nCandidates:\n{lines}")
        user = "\n\n".join(parts) + "\n\nAnswer with <question number>|<line number>|<verbatim words> per question."
        text = _llm_generate(RERANK_SYSTEM, user, 40 * len(items) + 20, raw_cache)
        for line in text.splitlines():
            m = re.match(r"\s*(\d+)\s*\|\s*\[?(\d+)\]?\s*\|?\s*(.*)$", line)
            if not m:
                continue
            qi, li = int(m.group(1)) - 1, int(m.group(2))
            allowed = dict(items).get(qi)
            if allowed and li in allowed:
                out[qi] = (li, m.group(3).strip())
    except Exception:
        logger.exception("rerank failed")
    return out


def answer_all(segments: Sequence[Dict], questions: Sequence[str], deadline: Optional[float] = None,
               raw_cache: Optional[Dict[str, str]] = None):
    """Return (answers, spans) from one LLM call.  Never raises."""
    n = len(questions)
    if SPAN_STRATEGY.get("mode") == "units":
        try:
            import units_mode
            return units_mode.answer_all_units(segments, questions, deadline, raw_cache, SPAN_STRATEGY)
        except Exception:
            logger.exception("units mode failed; falling back to quote mode")
    if deadline is not None and time.perf_counter() > deadline - MAIN_RESERVE:
        logger.warning("no time for the LLM (%.1fs left); lexical answers", deadline - time.perf_counter())
        parsed = [(None, None, "") for _ in range(n)]
        return _finish(segments, questions, parsed, {})
    try:
        import hashlib
        user = build_prompt(segments, questions)
        key = hashlib.sha1((SYSTEM_PROMPT + "\n" + user + LLM_REPO).encode()).hexdigest()
        t0 = time.perf_counter()
        if raw_cache is not None and key in raw_cache:
            text = raw_cache[key]
        else:
            text = _llm_generate(SYSTEM_PROMPT, user, MAX_NEW_TOKENS, None)
            if raw_cache is not None:
                raw_cache[key] = text
        logger.info("llm answered %d questions in %.1fs", n, time.perf_counter() - t0)
        LAST_RAW["text"] = text
        parsed = parse_answers(text, n)
    except Exception:
        logger.exception("LLM failed; using lexical fallback")
        parsed = [(None, None, "") for _ in range(n)]

    can_rerank = SPAN_STRATEGY.get("rerank") and (deadline is None or time.perf_counter() < deadline - RERANK_RESERVE)
    if SPAN_STRATEGY.get("rerank") and not can_rerank:
        logger.warning("skipping re-rank pass: %.1fs left", (deadline - time.perf_counter()) if deadline else -1)
    chosen = rerank_evidence(segments, questions, parsed, raw_cache) if can_rerank else {}
    answers, spans = _finish(segments, questions, parsed, chosen)
    if SPAN_STRATEGY.get("ce_ens") and (deadline is None or time.perf_counter() < deadline - 4.0):
        spans = _ce_ensemble(segments, questions, answers, spans)
    return answers, spans


def _overlap(a: Span, b: Span) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _ce_ensemble(segments: Sequence[Dict], questions: Sequence[str], answers: Sequence[bool],
                 spans: List[Optional[Span]]) -> List[Optional[Span]]:
    """Second opinion from the learned cross-encoder (reranker.py) over silence units.

    The LLM's quote and the cross-encoder pick the right passage about equally
    often (~68 %) but on different questions.  Rule (fitted on held-out folds):
    keep the LLM span when it overlaps one of the cross-encoder's top ``ce_top``
    units; otherwise, when the cross-encoder is confident (z-score margin
    between its best and second unit >= ``ce_margin``) or the LLM gave no span,
    take the cross-encoder's best unit grown to neighbours within ``ce_grow``.
    Never raises; ~0.3 s per conversation on the Mac.
    """
    cfg = SPAN_STRATEGY
    try:
        import numpy as np
        import reranker
        import units_mode
        units = units_mode.build_units(segments)
        if len(units) < 3:
            return spans
        k = int(cfg.get("ce_top", 2))
        margin = float(cfg.get("ce_margin", 1.0))
        grow = float(cfg.get("ce_grow", 0.5))
        texts = [u["text"] for u in units]
        out = list(spans)
        for qi, (ans, span) in enumerate(zip(answers, spans)):
            if not ans:
                continue
            sc = reranker.score(questions[qi], texts).astype(np.float64)
            z = (sc - sc.mean()) / (sc.std() + 1e-6)
            top = np.argsort(-z)
            if span is not None and any(_overlap(span, (units[t]["start"], units[t]["end"])) for t in top[:k]):
                continue
            zs = sorted(z.tolist(), reverse=True)
            if span is None or zs[0] - zs[1] >= margin:
                i = int(top[0])
                lo = hi = i
                while lo - 1 >= 0 and z[lo - 1] >= z[i] - grow:
                    lo -= 1
                while hi + 1 < len(z) and z[hi + 1] >= z[i] - grow:
                    hi += 1
                out[qi] = (round(units[lo]["start"], 2), round(units[hi]["end"], 2))
        return out
    except Exception:
        logger.exception("cross-encoder ensemble failed; keeping LLM spans")
        return spans


def _finish(segments, questions, parsed, chosen):
    answers: List[bool] = []
    spans: List[Optional[Span]] = []
    for i, (ans, seg, quote) in enumerate(parsed):
        if ans is None:
            ans, ids = lexical_fallback(segments, questions[i])
            seg, quote = (ids[0] if ids else None), ""
        if ans and i in chosen:
            new_seg, new_quote = chosen[i]
            if SPAN_STRATEGY.get("rerank_quote", 1) and new_quote:
                seg, quote = new_seg, new_quote
            elif new_seg != seg:
                # The re-ranker moved us to another line: quote no longer applies.
                seg, quote = new_seg, ""
        span = spans_for(segments, questions[i], quote, seg) if ans else None
        answers.append(bool(ans))
        spans.append(span)
    return answers, spans


def spans_for(segments: Sequence[Dict], question: str, quote: str, seg: Optional[int]) -> Optional[Span]:
    """Turn a (quote, line hint) into a span according to SPAN_STRATEGY."""
    mode = SPAN_STRATEGY.get("mode", "quote")
    pad = float(SPAN_STRATEGY.get("pad", 0.0))
    span = None
    if mode in ("embed", "embed_hybrid"):
        ranked = embed_rank(segments, question)          # [(seg_idx, sim)] best first
        if ranked:
            best_i = ranked[0][0]
            q_span = locate_quote(segments, quote, seg) if mode == "embed_hybrid" else None
            top = {i for i, _ in ranked[:int(SPAN_STRATEGY.get("embed_top", 2))]}
            if q_span is not None and any(segments[i]["end"] > q_span[0] and segments[i]["start"] < q_span[1] for i in top):
                span = q_span
            else:
                sub = lexical_span([segments[best_i]], question, return_score=True)
                if sub is not None and sub[1] >= float(SPAN_STRATEGY.get("embed_sub_min", 0.35)):
                    span = sub[0]
                else:
                    span = (segments[best_i]["start"], segments[best_i]["end"])
    if mode == "lexical":
        span = lexical_span(segments, question)
    elif mode == "hybrid":
        lex = lexical_span(segments, question, return_score=True)
        if lex is not None and lex[1] >= float(SPAN_STRATEGY.get("hybrid_min", 0.5)):
            span = lex[0]
        else:
            span = locate_quote(segments, quote, seg)
    if mode in ("quote", "quote_seg", "quote_pair"):
        span = locate_quote(segments, quote, seg)
    if span is not None and mode == "quote_seg":
        # Expand to the boundaries of the segment(s) the quote lives in.
        ids = [i for i, sg in enumerate(segments) if sg["end"] > span[0] and sg["start"] < span[1]]
        if ids:
            span = (segments[ids[0]]["start"], segments[ids[-1]]["end"])
    if span is not None and mode == "quote_pair":
        # If the quote is a single short turn, add the neighbouring turn that
        # completes the exchange (gold spans often cover question + answer).
        ids = [i for i, sg in enumerate(segments) if sg["end"] > span[0] and sg["start"] < span[1]]
        if len(ids) == 1:
            i = ids[0]
            nxt = segments[i + 1] if i + 1 < len(segments) else None
            if nxt is not None and (nxt["end"] - nxt["start"]) <= 2.0 and nxt["start"] - span[1] < 1.2:
                span = (span[0], nxt["end"])
    if span is None and seg is not None and 0 <= seg < len(segments):
        sub = lexical_span([segments[seg]], question, return_score=True)
        if sub is not None and sub[1] >= float(SPAN_STRATEGY.get("seg_sub_min", 0.35)):
            span = sub[0]
        else:
            span = span_from_segments(segments, [seg])
    if span is None:
        _, ids = lexical_fallback(segments, question)
        span = span_from_segments(segments, ids)
    if span is not None and pad:
        span = (round(max(0.0, span[0] - pad), 2), round(span[1] + pad, 2))
    if span is not None and SPAN_STRATEGY.get("snap", 1) and segments:
        sil = segments[0].get("_silences")
        if sil:
            span = snap_span(tuple(span), sil,
                             float(SPAN_STRATEGY.get("snap_start_tol", 0.6)),
                             float(SPAN_STRATEGY.get("snap_end_tol", 0.3)))
    return span


# --------------------------------------------------------------------------- #
# Evidence spans and fallbacks
# --------------------------------------------------------------------------- #

def _norm_words(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower())]


def locate_quote(segments: Sequence[Dict], quote: str, hint_seg: Optional[int], min_ratio: float = 0.6) -> Optional[Span]:
    """Find a verbatim quote in the word-timestamped transcript and return its span.

    Fuzzy: the quote is matched as a token sequence against a window of the
    same length sliding over the transcript words, and the best window wins if
    it agrees on >= min_ratio of the tokens.  Windows near ``hint_seg`` are
    preferred when scores tie.
    """
    q = _norm_words(quote)
    if len(q) < 2:
        return None
    words = []
    for si, s in enumerate(segments):
        for w in s.get("words", []):
            toks = _norm_words(w["word"])
            for t in toks:
                words.append((t, w["start"], w["end"], si))
    if not words:
        return None
    L = len(q)
    best, best_i = 0.0, -1
    import difflib
    for i in range(0, max(1, len(words) - L + 1)):
        window = [w[0] for w in words[i:i + L]]
        ratio = difflib.SequenceMatcher(None, q, window, autojunk=False).ratio()
        if hint_seg is not None and abs(words[i][3] - hint_seg) <= 1:
            ratio += 0.02
        if ratio > best:
            best, best_i = ratio, i
    if best_i < 0 or best < min_ratio:
        return None
    start = words[best_i][1]
    end = words[min(len(words) - 1, best_i + L - 1)][2]
    if end <= start:
        end = start + 0.5
    return (round(start, 2), round(end, 2))


def span_from_segments(segments: Sequence[Dict], ids: Sequence[int]) -> Optional[Span]:
    ids = sorted({i for i in ids if 0 <= i < len(segments)})
    if not ids:
        return None
    # Keep the citation contiguous and short: gold spans are a few seconds.
    start = segments[ids[0]]["start"]
    end = segments[ids[-1]]["end"]
    if end - start > MAX_SPAN_SECONDS and len(ids) > 1:
        # Take the longest run of consecutive ids, then trim to the first two.
        best_run, run = [ids[0]], [ids[0]]
        for a, b in zip(ids, ids[1:]):
            run = run + [b] if b == a + 1 else [b]
            if len(run) > len(best_run):
                best_run = run
        best_run = best_run[:2]
        start, end = segments[best_run[0]]["start"], segments[best_run[-1]]["end"]
    if end <= start:
        end = start + 0.5
    return (round(start, 2), round(end, 2))


_WORD_RE = re.compile(r"[a-z0-9.]+")
_STOP = set("the a an is are was were be been being do does did has have had will would should could can may might "
            "of to in on at for with by from as about into over after before during than that this these those it its "
            "there their they them he she his her him you your we our i me my any some no not any there s t did "
            "patient doctor mention mentioned discussed discussion right isn didn wasn weren".split())


def _tokens(text: str) -> set:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOP and len(w) > 1}


_EMB = {"model": None}
EMBED_REPO = os.environ.get("EMBED_REPO", "sentence-transformers/all-MiniLM-L6-v2")


def load_embedder():
    if _EMB["model"] is None:
        from sentence_transformers import SentenceTransformer
        _EMB["model"] = SentenceTransformer(EMBED_REPO, device="cpu")
    return _EMB["model"]


_SEG_EMB_CACHE: Dict[int, Tuple[int, "np.ndarray"]] = {}


def embed_rank(segments: Sequence[Dict], question: str) -> List[Tuple[int, float]]:
    """Segments ranked by cosine similarity to the question (best first).

    Segment embeddings are cached per transcript object so the ten questions
    of one conversation share a single encode call.
    """
    try:
        model = load_embedder()
        key = id(segments)
        cached = _SEG_EMB_CACHE.get(key)
        if cached is None or cached[0] != len(segments):
            texts = [sg["text"] for sg in segments]
            emb = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
            _SEG_EMB_CACHE.clear()
            _SEG_EMB_CACHE[key] = (len(segments), emb)
            cached = _SEG_EMB_CACHE[key]
        q = model.encode([question], normalize_embeddings=True, show_progress_bar=False)[0]
        sims = cached[1] @ q
        order = np.argsort(-sims)
        return [(int(i), float(sims[i])) for i in order]
    except Exception:
        logger.exception("embedding rank failed")
        return []


_STEM_RE = re.compile(r"(ing|ed|es|s|ly)$")


def _stem(w: str) -> str:
    return _STEM_RE.sub("", w) if len(w) > 4 else w


_QSTOP = _STOP | set("been being have has had does did do is are was were will would should could can may might "
                     "there any also still yet ever already just really actually about than then only both each "
                     "one two three four five six seven eight nine ten".split())


def lexical_span(segments: Sequence[Dict], question: str, return_score: bool = False,
                 min_words: int = 3, max_words: int = 14):
    """Best-matching word window for a question, from word timestamps.

    Score = sum of idf-like weights of question content words present in the
    window, normalised by the total question weight, with a mild penalty for
    long windows.  Numbers and drug-like tokens (rare words) weigh the most.
    """
    words = []
    for si, sg in enumerate(segments):
        for w in sg.get("words", []):
            for t in _norm_words(w["word"]):
                words.append((t, _stem(t), w["start"], w["end"]))
    if not words:
        return None
    q_tokens = [t for t in _norm_words(question) if t not in _QSTOP and len(t) > 1]
    q_stems = {_stem(t) for t in q_tokens}
    if not q_stems:
        return None
    # document frequency over the transcript, for idf-ish weights
    from collections import Counter
    df = Counter(st for _, st, _, _ in words)
    n_words = len(words)
    weight = {st: 1.0 + math.log((n_words + 1) / (df.get(st, 0) + 1)) for st in q_stems}
    total = sum(weight.values())
    best, best_span = 0.0, None
    stems = [st for _, st, _, _ in words]
    for i in range(n_words):
        seen = set()
        acc = 0.0
        for j in range(i, min(n_words, i + max_words)):
            st = stems[j]
            if st in q_stems and st not in seen:
                seen.add(st)
                acc += weight[st]
            L = j - i + 1
            if L < min_words:
                continue
            score = acc / total - 0.004 * L
            if score > best:
                best, best_span = score, (words[i][2], words[j][3])
    if best_span is None:
        return None
    span = (round(best_span[0], 2), round(max(best_span[1], best_span[0] + 0.3), 2))
    return (span, best) if return_score else span


def lexical_fallback(segments: Sequence[Dict], question: str) -> Tuple[bool, List[int]]:
    """Best-overlap segment pair for a question; yes only on strong overlap."""
    q = _tokens(question)
    if not q or not segments:
        return False, []
    best, best_i = 0.0, -1
    for i, s in enumerate(segments):
        window = _tokens(s["text"]) | (_tokens(segments[i + 1]["text"]) if i + 1 < len(segments) else set())
        score = len(q & window) / len(q)
        if score > best:
            best, best_i = score, i
    if best_i < 0:
        return False, []
    ids = [best_i] + ([best_i + 1] if best_i + 1 < len(segments) else [])
    return best >= 0.6, ids
