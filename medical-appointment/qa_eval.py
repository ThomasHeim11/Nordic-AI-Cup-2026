"""Offline evaluation of the answering half against question_train.csv.

Transcripts are cached in cache/transcripts/<transcript_id>.json so the LLM
prompt can be iterated without paying for ASR every time.

    python qa_eval.py --transcribe-only        # fill the cache
    python qa_eval.py --limit 10               # score the first 10 conversations
"""
import argparse, json, os, statistics, sys, time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache", "transcripts")


def cached_transcript(key: str):
    """key may be a transcript id ("sample_4") or an audio filename."""
    import pipeline
    from utils import load_sample_audio, audio_filename_for_transcript
    os.makedirs(CACHE, exist_ok=True)
    transcript_id = key[len("conversation_"):] if key.startswith("conversation_") else key
    transcript_id = transcript_id[:-4] if transcript_id.endswith(".mp3") else transcript_id
    path = os.path.join(CACHE, f"{transcript_id}.json")
    if os.path.exists(path):
        with open(path) as f:
            segs = json.load(f)
        if segs and "_silences" not in segs[0]:
            from faster_whisper.audio import decode_audio
            audio = decode_audio(os.path.join(HERE, "data", "audio", audio_filename_for_transcript(transcript_id)), sampling_rate=16000)
            segs[0]["_silences"] = pipeline.silence_intervals(audio)
        return segs
    t0 = time.perf_counter()
    segs = pipeline.transcribe(load_sample_audio(audio_filename_for_transcript(transcript_id)))
    with open(path, "w") as f:
        json.dump(segs, f)
    print(f"  transcribed {transcript_id}: {len(segs)} segments in {time.perf_counter() - t0:.1f}s", flush=True)
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--transcribe-only", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--mode", default="quote", help="span strategy: quote | quote_seg | quote_pair")
    ap.add_argument("--pad", type=float, default=0.0)
    ap.add_argument("--rerank", type=int, default=0)
    ap.add_argument("--seg-sub-min", type=float, default=0.35)
    ap.add_argument("--rerank-quote", type=int, default=0)
    ap.add_argument("--ce-weight", type=float, default=0.0, help="units mode: use the cross-encoder re-ranker")
    ap.add_argument("--llm-bonus", type=float, default=1.5)
    ap.add_argument("--rerank-bonus", type=float, default=1.0)
    ap.add_argument("--unit-gap", type=float, default=0.2)
    ap.add_argument("--snap", type=int, default=1)
    ap.add_argument("--snap-start-tol", type=float, default=0.6)
    ap.add_argument("--snap-end-tol", type=float, default=0.3)
    ap.add_argument("--ce-ens", type=int, default=1, help="cross-encoder second opinion on the evidence span (pipeline._ce_ensemble)")
    ap.add_argument("--ce-top", type=int, default=2)
    ap.add_argument("--ce-margin", type=float, default=1.0)
    ap.add_argument("--ce-grow", type=float, default=0.5)
    ap.add_argument("--rerank-ce-cands", type=int, default=0, help="experiment: add the cross-encoder's top-N lines to the re-rank candidates")
    ap.add_argument("--ce-lex-vote", type=int, default=0, help="experiment: lexical matcher as a third voter in the span ensemble")
    a = ap.parse_args()
    sys.path.insert(0, HERE)
    import pipeline
    from utils import group_questions_by_conversation, gold_evidence, temporal_iou

    pipeline.SPAN_STRATEGY.update({"mode": a.mode, "pad": a.pad, "rerank": a.rerank, "seg_sub_min": a.seg_sub_min, "rerank_quote": a.rerank_quote, "snap": a.snap, "snap_start_tol": a.snap_start_tol, "snap_end_tol": a.snap_end_tol,
                                   "ce_weight": a.ce_weight, "llm_bonus": a.llm_bonus, "rerank_bonus": a.rerank_bonus, "unit_gap": a.unit_gap,
                                   "ce_ens": a.ce_ens, "ce_top": a.ce_top, "ce_margin": a.ce_margin, "ce_grow": a.ce_grow,
                                   "rerank_ce_cands": a.rerank_ce_cands, "ce_lex_vote": a.ce_lex_vote})
    raw_path = os.path.join(HERE, "cache", "llm_raw.json")
    raw_cache = json.load(open(raw_path)) if os.path.exists(raw_path) else {}
    convs = group_questions_by_conversation()
    if a.limit:
        convs = convs[:a.limit]
    if a.transcribe_only:
        for tid, _ in convs:
            cached_transcript(tid)
        return

    correct = defaultdict(list)
    tious = []
    no_span = 0
    yes_pred_tious = []
    t_llm = []
    for tid, rows in convs:
        segs = cached_transcript(tid)
        qs = [r["question"] for r in rows]
        t0 = time.perf_counter()
        answers, spans = pipeline.answer_all(segs, qs, raw_cache=raw_cache)
        t_llm.append(time.perf_counter() - t0)
        with open(raw_path, "w") as f:
            json.dump(raw_cache, f)
        os.makedirs(os.path.join(HERE, "cache", "llm_out"), exist_ok=True)
        with open(os.path.join(HERE, "cache", "llm_out", f"{tid}.txt"), "w") as f:
            f.write(pipeline.LAST_RAW["text"])
        for r, ans, span in zip(rows, answers, spans):
            gold = r["answer"] == "yes"
            correct[r["question_type"]].append(ans == gold)
            if gold:
                g = gold_evidence(r)
                t = temporal_iou(g, span) if span else 0.0
                tious.append(t)
                if span is None:
                    no_span += 1
                if ans:
                    yes_pred_tious.append(t)
            if a.verbose and (ans != gold or (gold and (span is None or temporal_iou(gold_evidence(r), span) < 0.3))):
                print(f"  [{tid}] {r['question_type']:13s} gold={r['answer']:3s} pred={'yes' if ans else 'no':3s} "
                      f"span={span} gold_span={gold_evidence(r)} :: {r['question']}")
        print(f"{tid}: llm {t_llm[-1]:.1f}s", flush=True)

    all_correct = [c for v in correct.values() for c in v]
    acc = statistics.mean(all_correct)
    mt = statistics.mean(tious) if tious else 0.0
    print("\nAccuracy by type")
    for k, v in correct.items():
        print(f"  {k:14s} {statistics.mean(v):.3f} ({sum(v)}/{len(v)})")
    print(f"Accuracy {acc:.3f} | mean tIoU {mt:.3f} (n={len(tious)}, no span {no_span}) "
          f"| tIoU when yes {statistics.mean(yes_pred_tious) if yes_pred_tious else 0:.3f}")
    print(f"SCORE 0.4*acc + 0.6*tIoU = {0.4 * acc + 0.6 * mt:.3f}   (mode={a.mode} pad={a.pad} rerank={a.rerank} seg_sub_min={a.seg_sub_min})")
    print(f"LLM time per conversation: mean {statistics.mean(t_llm):.1f}s max {max(t_llm):.1f}s")


if __name__ == "__main__":
    main()
