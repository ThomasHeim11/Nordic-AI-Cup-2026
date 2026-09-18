"""Learned evidence re-ranker: a cross-encoder scoring (question, utterance) pairs.

We have 195 gold spans in question_train.csv.  Every utterance overlapping a
gold span is a positive for its question; every other utterance of the same
conversation is a negative.  A small MS-MARCO cross-encoder is fine-tuned on
those pairs -- it learns the annotators' convention of *which* utterance
counts as evidence, which the general-purpose LLM only approximates.

    python reranker.py cv                # leave-conversations-out CV (5 folds), honest numbers
    python reranker.py train             # train on all 39 conversations -> models/reranker
    (pipeline)  reranker.score(question, [unit texts]) -> np.ndarray of scores
"""
import csv
import json
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Sequence

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models", "reranker")
BASE = os.environ.get("RERANK_BASE", "cross-encoder/ms-marco-MiniLM-L-6-v2")
CONTEXT = os.environ.get("RERANK_CONTEXT", "0") == "1"   # score "prev || UNIT || next" so one-word units are not orphaned
EPOCHS = int(os.environ.get("RERANK_EPOCHS", "6"))


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def passage(units: Sequence[Dict], i: int) -> str:
    if not CONTEXT:
        return units[i]["text"]
    prev = units[i - 1]["text"] if i > 0 else ""
    nxt = units[i + 1]["text"] if i + 1 < len(units) else ""
    return f"{prev} || {units[i]['text']} || {nxt}"


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def load_conversations():
    """[(transcript_id, units, [(question, gold_span or None)])] from cached transcripts."""
    sys.path.insert(0, HERE)
    import pipeline, units_mode
    from faster_whisper.audio import decode_audio
    from utils import gold_evidence
    rows = list(csv.DictReader(open(os.path.join(HERE, "data", "question_train.csv"))))
    by = defaultdict(list)
    for r in rows:
        by[r["transcript_id"]].append(r)
    out = []
    for tid, rs in by.items():
        path = os.path.join(HERE, "cache", "transcripts", f"{tid}.json")
        if not os.path.exists(path):
            continue
        segs = json.load(open(path))
        if segs and "_silences" not in segs[0]:
            segs[0]["_silences"] = pipeline.silence_intervals(
                decode_audio(os.path.join(HERE, "data", "audio", f"conversation_{tid}.mp3"), sampling_rate=16000))
        units = units_mode.build_units(segs)
        qs = [(r["question"], gold_evidence(r)) for r in rs]
        out.append((tid, units, qs))
    return out


def gold_units(units: Sequence[Dict], span, min_iou=0.3) -> List[int]:
    from utils import temporal_iou
    ids = []
    for i, u in enumerate(units):
        inside = u["start"] >= span[0] - 0.25 and u["end"] <= span[1] + 0.25
        if inside or temporal_iou(span, (u["start"], u["end"])) >= min_iou:
            ids.append(i)
    return ids


def make_examples(convs, neg_per_q: int = 20, seed: int = 0):
    from sentence_transformers import InputExample
    rng = random.Random(seed)
    ex = []
    for tid, units, qs in convs:
        for q, span in qs:
            if span is None:
                continue
            pos = set(gold_units(units, span))
            if not pos:
                continue
            for i in pos:
                ex.append(InputExample(texts=[q, passage(units, i)], label=1.0))
            negs = [i for i in range(len(units)) if i not in pos]
            rng.shuffle(negs)
            for i in negs[:neg_per_q]:
                ex.append(InputExample(texts=[q, passage(units, i)], label=0.0))
    return ex


# --------------------------------------------------------------------------- #
# Train / evaluate
# --------------------------------------------------------------------------- #

def train_model(convs, out_dir: str, epochs: int = EPOCHS, batch: int = 32):
    from sentence_transformers import CrossEncoder
    from torch.utils.data import DataLoader
    ex = make_examples(convs)
    model = CrossEncoder(BASE, num_labels=1, max_length=256, device=_device())
    loader = DataLoader(ex, shuffle=True, batch_size=batch)
    model.fit(train_dataloader=loader, epochs=epochs, warmup_steps=int(0.1 * len(loader) * epochs),
              optimizer_params={"lr": 2e-5}, show_progress_bar=False)
    model.save(out_dir)
    return model


def evaluate(model, convs):
    """Selection hit rate and tIoU of the argmax unit, over positive questions."""
    from utils import temporal_iou
    hits, tious, n = 0, [], 0
    for tid, units, qs in convs:
        texts = [passage(units, i) for i in range(len(units))]
        for q, span in qs:
            if span is None:
                continue
            sc = model.predict([[q, t] for t in texts], show_progress_bar=False)
            best = int(np.argmax(sc))
            n += 1
            hits += best in set(gold_units(units, span))
            tious.append(temporal_iou(span, (units[best]["start"], units[best]["end"])))
    return hits / max(n, 1), float(np.mean(tious)) if tious else 0.0, n


def cv(folds: int = 5):
    convs = load_conversations()
    rng = random.Random(0)
    order = list(range(len(convs)))
    rng.shuffle(order)
    hit_all, tiou_all, n_all = [], [], 0
    for f in range(folds):
        test_idx = set(order[f::folds])
        train = [c for i, c in enumerate(convs) if i not in test_idx]
        test = [c for i, c in enumerate(convs) if i in test_idx]
        model = train_model(train, os.path.join(HERE, "models", f"reranker_cv{f}"))
        hit, tiou, n = evaluate(model, test)
        print(f"fold {f}: test conversations {len(test)} | positive questions {n} | argmax-in-gold {hit:.3f} | tIoU of argmax unit {tiou:.3f}", flush=True)
        hit_all.append(hit * n); tiou_all.append(tiou * n); n_all += n
    print(f"CV over {n_all} positive questions: selection hit {sum(hit_all)/n_all:.3f} | single-unit tIoU {sum(tiou_all)/n_all:.3f}")


_MODEL = {"m": None}


def score(question: str, unit_texts: Sequence[str]) -> np.ndarray:
    """Scores for every unit (higher = more likely the evidence).  Units are given as texts
    in order; context passages are rebuilt here so callers only pass the texts."""
    if _MODEL["m"] is None:
        from sentence_transformers import CrossEncoder
        _MODEL["m"] = CrossEncoder(MODEL_DIR, max_length=256, device=_device())
    units = [{"text": t} for t in unit_texts]
    texts = [passage(units, i) for i in range(len(units))]
    return np.asarray(_MODEL["m"].predict([[question, t] for t in texts], show_progress_bar=False), dtype=np.float32)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "cv"
    if cmd == "cv":
        cv()
    elif cmd == "train":
        m = train_model(load_conversations(), MODEL_DIR)
        hit, tiou, n = evaluate(m, load_conversations())
        print(f"trained on all; resubstitution (optimistic): hit {hit:.3f} tIoU {tiou:.3f} (n={n}) -> {MODEL_DIR}")
