# Nordic AI Cup 2026 — competition facts

## Timeline
- Start: 2026-09-17 10:00 CEST
- **Deadline: 2026-09-20 16:00 CEST (UTC+2)**
- Top-5 teams must submit training code + trained models by 2026-09-20 20:00 CEST.

## Links
- Repo (upstream): https://github.com/amboltio/Nordic-AI-Cup-2026
- Submission + scoreboard: https://cases.nordicaicup.com/
- Discord: https://discord.gg/z6nnJ6Dsq
- Contact: nordicaicup@ambolt.io

## Eligibility
Students at a university in DK/NO/SE/FI/IS. Max 4 (BSc/MSc) or 2 (PhD) per team.
**All members must represent the same country** — we are 🇳🇴 Norway.

## How submission works
Every challenge = you host a **FastAPI endpoint reachable from the public internet**.
You submit `http://<host>:<port>/predict` — the URL is used **exactly as given, path included**.

Three buttons on cases.nordicaicup.com:
- **Verify** — single request, format check only. Generous timeout (30 s drone). Not a speed check.
- **Queue validation attempt** — unlimited, runs on the *validation* set, shows on scoreboard.
- **Evaluation attempt** — **ONE per challenge, ever**. Different (harder/larger) dataset. This is the judged score.

=> Validate a lot. Only press Evaluate when the server is stable and warm.

## Scoring (overall)
Per challenge you get an F1-style rank score:
25 / 18 / 15 / 12 / 10 / 8 / 6 / 4 / 2 / 1 points for places 1–10, then 1→0 for 11+.
Total = sum over the three challenges. **Submitting anything at all in all three beats
being excellent in one** — a mediocre entry still ranks and still scores points.

## Hard rules
- **No cloud APIs at inference time.** No OpenAI/Anthropic/Google/AWS calls inside /predict.
  You MAY use any cloud API/service while developing and training.
- Pretrained models are allowed and encouraged.
- You may gather/generate your own extra training data.
- No servers provided for training — own hardware / Colab / Azure for Students.

## Our hardware
- MacBook Pro, Apple M1 Pro, 14-core GPU, 16 GB unified memory, Python 3.11.1.
- No CUDA. Torch `mps` backend works; `faster-whisper` (CTranslate2) is **CPU-only** on macOS.
  Apple-Silicon-friendly ASR: `mlx-whisper`, `whisper.cpp` (Metal), or `lightning-whisper-mlx`.
- Open question: do we have an NVIDIA box / Azure-for-Students VM for hosting + training?
  Hosting is the bigger risk — the endpoint must be publicly reachable.
