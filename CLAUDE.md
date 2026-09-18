# Nordic AI Cup 2026 — 🇳🇴 Norway team

Deadline: **2026-09-20 16:00 CEST**. Three challenges, each = a public FastAPI `/predict` endpoint.

Reference notes (read these before working on a challenge — they are the READMEs condensed):
- `.claude/competition.md` — rules, submission, scoring, hardware
- `.claude/challenge-1-survival.md` — Survival Simulator (port 9052)
- `.claude/challenge-2-drone.md` — Drone Flyby (port 9053)
- `.claude/challenge-3-medical.md` — Medical Appointment (port 9054)
- `.claude/status.md` — running log: what is done, what is next, measured scores

The project root is a git clone of amboltio/Nordic-AI-Cup-2026 (templates + our code side by side).
Our code goes in that tree so the local evaluators work unchanged.

## Goal: win. We are benchmark-maxing all three challenges.
Rank points per challenge: 1st 25 · 2nd 18 · 3rd 15 · 4th 12 · 5th 10 · 6th 8 · 7th 6 · 8th 4 ·
9th 2 · 10th 1 · 11th+ <1. Total = sum. Every rank step near the top is worth more than the
one below it, so the target for each challenge is **1st**, not "good enough". Validation
attempts are free and unlimited: measure on the real scoreboard, not only locally.
Compute: Mac M1 Pro (serving medical), Simula cluster n009 (128 ARM cores + A40 46 GB) for
training/tuning and for serving drone + survival. Cluster access is via the user's terminal.

## Standing rules
- **No cloud API calls inside `/predict`.** Anything goes during development/training.
- Only **one evaluation attempt per challenge, ever**. Validate as often as you like first.
- Points are by rank (25/18/15/12/10/8/6/4/2/1). A weak submission in all three beats a
  strong submission in one — always ship something for every challenge.
- Never let an endpoint raise. Catch everything and return a valid, well-shaped response.
- Warm models up at import time.
