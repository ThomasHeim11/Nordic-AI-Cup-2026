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

## Standing rules
- **No cloud API calls inside `/predict`.** Anything goes during development/training.
- Only **one evaluation attempt per challenge, ever**. Validate as often as you like first.
- Points are by rank (25/18/15/12/10/8/6/4/2/1). A weak submission in all three beats a
  strong submission in one — always ship something for every challenge.
- Never let an endpoint raise. Catch everything and return a valid, well-shaped response.
- Warm models up at import time.
