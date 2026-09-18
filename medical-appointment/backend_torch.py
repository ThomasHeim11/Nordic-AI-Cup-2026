"""CUDA/CPU backend for the medical pipeline, via HuggingFace transformers.

Mirrors the two MLX calls the pipeline makes on the Mac:

    transcribe_words(audio16k) -> whisper result dict with segments + word timestamps
    chat(system, user, max_new_tokens) -> generated text

Select with MEDICAL_BACKEND=torch.  Models via env:
    ASR_HF_REPO  (default openai/whisper-large-v3-turbo)
    LLM_HF_REPO  (default Qwen/Qwen2.5-32B-Instruct)
"""
import logging
import os
import time
from typing import Dict, List

import numpy as np
import torch

logger = logging.getLogger(__name__)

ASR_HF_REPO = os.environ.get("ASR_HF_REPO", "openai/whisper-large-v3-turbo")
LLM_HF_REPO = os.environ.get("LLM_HF_REPO", "Qwen/Qwen2.5-32B-Instruct")
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

_ASR = {"pipe": None}
_LLM = {"model": None, "tok": None}


def load_asr():
    if _ASR["pipe"] is None:
        from transformers import pipeline
        t0 = time.perf_counter()
        _ASR["pipe"] = pipeline("automatic-speech-recognition", model=ASR_HF_REPO, torch_dtype=DTYPE,
                                device=DEVICE, chunk_length_s=30, stride_length_s=5)
        logger.info("ASR %s loaded on %s in %.1fs", ASR_HF_REPO, DEVICE, time.perf_counter() - t0)
    return _ASR["pipe"]


def transcribe_words(audio: np.ndarray, sr: int = 16000) -> Dict:
    """Return {"segments": [{"start","end","text","words":[{"word","start","end"}]}]}.

    transformers gives word-level chunks; sentences are rebuilt from punctuation
    so the pipeline sees the same segment granularity as with whisper on MLX.
    """
    pipe = load_asr()
    gen = {} if ASR_HF_REPO.endswith(".en") else {"language": "en", "task": "transcribe"}
    out = pipe({"raw": audio.astype(np.float32), "sampling_rate": sr}, return_timestamps="word",
               generate_kwargs=gen)
    words = []
    last_end = 0.0
    for ch in out.get("chunks", []):
        w = ch["text"].strip()
        if not w:
            continue
        s, e = ch["timestamp"]
        s = float(s) if s is not None else last_end
        e = float(e) if e is not None else s + 0.3
        last_end = e
        words.append({"word": w, "start": s, "end": e})
    segments, cur = [], []
    for w in words:
        cur.append(w)
        if w["word"][-1:] in ".?!" and len(cur) >= 2 or len(cur) >= 30:
            segments.append(cur); cur = []
    if cur:
        segments.append(cur)
    return {"segments": [{"start": seg[0]["start"], "end": seg[-1]["end"],
                          "text": " ".join(w["word"] for w in seg), "words": seg} for seg in segments]}


def load_llm():
    if _LLM["model"] is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        t0 = time.perf_counter()
        tok = AutoTokenizer.from_pretrained(LLM_HF_REPO)
        model = AutoModelForCausalLM.from_pretrained(LLM_HF_REPO, torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
                                                     device_map="auto" if DEVICE == "cuda" else None)
        if DEVICE != "cuda":
            model = model.to(DEVICE)
        model.eval()
        _LLM["model"], _LLM["tok"] = model, tok
        logger.info("LLM %s loaded on %s in %.1fs", LLM_HF_REPO, DEVICE, time.perf_counter() - t0)
    return _LLM["model"], _LLM["tok"]


@torch.inference_mode()
def chat(system: str, user: str, max_new_tokens: int = 400) -> str:
    model, tok = load_llm()
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                     return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
