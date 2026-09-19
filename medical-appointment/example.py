"""Medical appointment endpoint: local whisper (MLX) + local Qwen (MLX).

Everything heavy is loaded and exercised at import time, because the first
request gets no grace period.  ``predict`` never raises: any failure degrades to
a well-formed guess, since a malformed body loses all ten questions.
"""

import logging
import time
from typing import List, Optional

from dtos import ASRQuestionRequestDto, ASRQuestionResponseDto
from utils import decode_audio

import pipeline

logger = logging.getLogger(__name__)

# Whole-attempt budget is 60 s per conversation on average; keep a margin.
# The pipeline drops the re-rank pass, then the LLM itself, as this runs out.
SOFT_BUDGET_SECONDS = 52.0

# Requests are serialized: the evaluator sends one at a time, but after a
# timeout it moves on while we may still be busy, and two LLMs on one GPU is
# how a single slow conversation turns into five.
import threading
_LOCK = threading.Lock()


def _warm_up() -> None:
    try:
        import numpy as np
        t0 = time.perf_counter()
        import mlx_whisper
        mlx_whisper.transcribe(np.zeros(16000 * 3, dtype=np.float32), path_or_hf_repo=pipeline.ASR_REPO,
                               language="en", condition_on_previous_text=False, fp16=True)
        logger.info("ASR warm in %.1fs", time.perf_counter() - t0)
    except Exception:
        logger.exception("ASR warm-up failed")
    try:
        t0 = time.perf_counter()
        pipeline.answer_all([{"start": 0.0, "end": 1.0, "text": "The dose is 100 mg daily."}],
                            ["Is the dose 100 mg daily?"])
        logger.info("LLM warm in %.1fs", time.perf_counter() - t0)
    except Exception:
        logger.exception("LLM warm-up failed")
    try:
        t0 = time.perf_counter()
        import reranker
        reranker.score("Is the dose 100 mg daily?", ["The dose is 100 mg daily.", "Take it with food.", "See you next month."])
        logger.info("cross-encoder warm in %.1fs", time.perf_counter() - t0)
    except Exception:
        logger.exception("cross-encoder warm-up failed (span ensemble will fall back to LLM spans)")


_warm_up()


def predict(request: ASRQuestionRequestDto) -> ASRQuestionResponseDto:
    with _LOCK:
        return _predict(request)


def _predict(request: ASRQuestionRequestDto) -> ASRQuestionResponseDto:
    t0 = time.perf_counter()
    n = len(request.questions)
    answers: List[bool] = [False] * n
    starts: List[Optional[float]] = [None] * n
    ends: List[Optional[float]] = [None] * n
    try:
        audio_bytes = decode_audio(request.audio_base64)
        segments = pipeline.transcribe(audio_bytes)
        t_asr = time.perf_counter() - t0
        logger.info("%s: %d segments in %.1fs", request.audio_filename, len(segments), t_asr)
        if segments:
            a, spans = pipeline.answer_all(segments, list(request.questions),
                                           deadline=t0 + SOFT_BUDGET_SECONDS)
            if len(a) == n and len(spans) == n:
                answers = [bool(x) for x in a]
                starts = [float(s[0]) if s else None for s in spans]
                ends = [float(s[1]) if s else None for s in spans]
    except Exception:
        logger.exception("predict failed for %s; returning guesses", request.audio_filename)

    # Belt and braces: the three lists must line up or the whole body is lost.
    answers = (answers + [False] * n)[:n]
    starts = (starts + [None] * n)[:n]
    ends = (ends + [None] * n)[:n]
    for i in range(n):
        if not answers[i] or starts[i] is None or ends[i] is None or ends[i] < starts[i]:
            starts[i] = ends[i] = None
    logger.info("%s answered in %.1fs (%s yes)", request.audio_filename,
                time.perf_counter() - t0, sum(answers))
    return ASRQuestionResponseDto(answers=answers, evidence_start=starts, evidence_end=ends)
