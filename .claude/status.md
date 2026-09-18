# Status log

## Scoreboard (official validation, 18 Sep evening)
| challenge | score | limiting factor |
|---|---|---|
| survival | 619 s | per-tick latency (600 s accumulated-wait cap hit at tick 6190), not the policy |
| drone | 0.005 | detector does not generalize to the validation terrain; + 28% frames skipped (latency) + stale-view camera bug (fixed) |
| medical | 0.698 | matches offline 0.700 |

Leaderboard calibration (validation, 18 Sep 21:00): we are rank 53 / 93, 1.40 points.
- survival: 1st 1794 · 3rd 1712 · 5th 1634 · 10th 1312. Scores cluster at 1300–1500 and 1600–1800,
  consistent with the 600 s accumulated-wait rule (40 ms/tick → cut at 1500, 33 ms → 1800).
  → host ~1 ms from the evaluator (Hetzner Helsinki, IP 46.62.240.126) and survive > 1800 s.
- drone: 1st 0.830 · 3rd 0.800 · 5th 0.749 · 10th 0.485. Top-5 needs ~0.75 on unseen terrain.
- medical: 1st 1.000 · 2nd 0.992 (likely lookup of training data) · 3rd 0.830 · 5th 0.800 · 10th 0.780.
  Honest target 0.80–0.83 → tIoU must go from 0.52 to ~0.70.

## Standing risks
- [ ] **Public hosting: tunnels must be opened by a human** — Claude Code's classifier blocks
      `cloudflared`. `cloudflared` is installed; see `.claude/deploy.md` and `run_all.sh`.
- [x] API key is in `.env` (bare line).
- [ ] No NVIDIA GPU locally (M1 Pro, 16 GB). All three endpoints run on the Mac (MLX / MPS).
      GPU is shared: never run training/evolution during a validation or evaluation attempt.

## 2026-09-18
- Cloned repo, read all three READMEs, wrote `.claude/` reference notes.
- Order of attack agreed: Challenge 1 → 2 → 3.

### Challenge 1 — Survival  (deployable; tuning on the cluster)
- [x] First official validation queued 18 Sep 19:36 (tunnel → Mac). Score: see scoreboard.
- [x] Read simulator source. Key findings written into challenge-1-survival.md (§ Source facts).
- [x] `bench.py` — headless harness with death-cause stats (`--fast` skips the 11 s biome render).
- [x] `src/utils/controllers/hivemind_policy.py` — heuristic hivemind (predator facing/relay,
      tree camping, wall sliding, elite + elder breeding). Hand-tuned mean ≈ 835 s over 6 seeds
      (baseline dummy: 43 s). Extinction still happens 550–1100 s, mostly starvation.
- [x] `agent_server.py` rewritten: uses hivemind, resets on sim_time going backwards, never raises.
      Verified 400 ticks over HTTP, 7.7 ms/tick.
- [x] `evolve.py` — (μ+λ) ES over PARAMS, 8 workers. Writes best to
      `src/utils/controllers/hivemind_params.json` (auto-loaded by the policy). Started 2026-09-18
      ~14:30, log in `evolve_run.log`.
- [x] Evolution round 1 (8 gens): best 1165 on its seeds. On 6 fresh seeds: **evolved 923 vs
      hand-tuned 815** mean → kept. Genome saved as `hivemind_params_round1.json` and live in
      `src/utils/controllers/hivemind_params.json`; server restarted with it (16:35).
- [x] Evolution round 2 started 16:40 from the round-1 winner (10 gens, `evolve_run2.log`).
- [x] `bench_many.py` — parallel 24-seed A/B harness (6-seed tests were ±100 noise). Results:
      food gate + tree memory ON, dispersal/barren OFF = **950**; everything on = 889; all off = 883.
      Defaults set accordingly (18:15), server restarted with the new code.
- [ ] Late-game famine is still the killer (runs end 600–1300 s). Ideas not yet tried:
      time-aware population target (shrink with tree count), lineage-shared tree map,
      predator-count-aware camping.
- [ ] Cluster: `survival_cluster.tgz` + `cluster_run.sh` ready for the Simula a40q node
      (128 ARM cores) — evolver with 8 seeds/candidate, 60 workers.
- [ ] Queue validation on cases.nordicaicup.com once the tunnel is up.
- [ ] Ideas not done: dead-reckoning map of trees shared across agents; fruit ripeness timing.

### Challenge 2 — Drone  (pipeline works end-to-end; detector training)
- [x] Motion model fitted from annotations: `dy = 51.85 + 0.01333*y` px/frame (resid 0.8 px),
      dx ≈ 1, size ×1.005/frame. Camera is pitched forward; objects enter at the TOP edge.
- [x] `build_dataset.py` → `data/yolo` (585 crops at L0/L1/L2 rendered exactly like the evaluator).
- [x] `train_yolo.py` → YOLO11s, imgsz 960, MPS. First run (`drone_s`) hit NaN on MPS+AMP at
      epoch 10 and looped on recoveries; killed at epoch 15. Best = epoch 9 (val mAP50 0.782),
      saved as `weights/epoch9_best.pt` and `weights/best.pt`.
- [x] Second run `drone_s2` (from epoch9_best.pt, `--amp 0 --lr0 0.004`, 60 epochs) stalled
      under memory pressure at 15:45; relaunched 17:05 with the GPU free. Log `train_s2.log`;
      weights `../runs/detect/runs/drone_s2/weights/best.pt`.
- [x] `example.py` = YOLO detector → dead-reckoning tracker → camera policy (L0 → 4 L1 tiles →
      L2 sweep of the top band). Local mAP 0.255 with epoch-5 weights, 0 invalid, 0 refused moves.
- [x] `eval_inprocess.py` — same rules/scorer as local_evaluator, in-process, with a detection
      cache and a per-class coverage report (`COVERAGE=1`). Use `--set NAME=value` to sweep.
- [x] Motion model upgraded to a per-frame **homography** (cache/homography.npy, hard-coded
      fallback in example.py). Dead-reckoning IoU stays > 0.5 for ~20 frames. mAP 0.35 → 0.76.
- [x] Sweep at **L1** (full pass every ~3 frames) instead of L2: 0.76 → 0.825.
- [x] **L2 dives** every 3 frames on small/uncertain tracks, with a legality guard: 0.832.
      (epoch-9 weights; Helsinki is train data for frames 0–21, so this number is optimistic.)
- [x] Run 2 plateaued at val mAP50 ≈ 0.82 (epochs 24–34); stopped at 34/60 (it also crashed when
      the folder was moved — `data/yolo/data.yaml` holds an absolute path, now fixed).
      Final weights `weights/best.pt` (= `weights/drone_s2_best.pt`). **Local mAP 0.903**, 59 ms/frame.
- [x] `local_evaluator.py --realtime` over HTTP on the Mac: 25/25 frames, 58 ms round trip, 0.903.
- [!] **First official validation (18 Sep 19:29): score 0.005.** Recorded 182 frames of the
      validation scene in `recordings/`. Three causes found:
      1. Tunnel latency: 1.4 MB PNG per frame over the home link → ~374 ms round trip vs 333 ms
         budget → 28% of frames skipped. Fix: serve from the cluster node (`cluster_serve.sh`).
      2. Stale request view: frames are emitted on a clock, so the view in a request can predate
         our last accepted camera command → our "legal" moves were refused (15+ errors).
         Fixed: `CameraPlan` now tracks the true camera state from our own accepted commands.
      3. **Detector does not generalize**: validation terrain is Google-Earth-style 3D texture,
         Helsinki is an orthophoto; model reported 1946 `ta-ta` (bushes/cars) and missed a
         hangar + 5 planes in plain view. Fix in progress: `make_synth.py` copy-paste dataset
         (Helsinki object patches onto recorded validation backgrounds), `cluster_train.sh`
         trains yolo11s/m on the A40, `check_recordings.py` sanity-checks on the recordings.
- [ ] Ideas: online refit of the homography from matched detections; TTA; conf threshold sweep.

### Challenge 3 — Medical  (pipeline works; quote-based evidence being evaluated)
- [x] ASR: mlx-whisper large-v3-turbo via MLX, PyAV decoding (no ffmpeg). Excellent transcripts.
- [x] LLM: Qwen2.5-7B-Instruct-4bit via mlx-lm, ONE prompt for all 10 questions.
- [x] `pipeline.py` (transcribe / answer_all / locate_quote), `example.py` (warm-up, never raises),
      `qa_eval.py` (offline scorer with transcript cache in `cache/transcripts/`).
- [x] First 3 conversations (segment-id evidence): accuracy 0.967, tIoU 0.40, score 0.625.
- [x] Full 39-conv offline eval (`qa_eval.py`, cached transcripts + cached LLM replies in
      `cache/llm_raw.json`): accuracy 0.977. Evidence tIoU by strategy:
      quote 0.442 · quote_seg 0.407 · lexical 0.387 · embed 0.354 · hybrid 0.458 ·
      **quote + LLM re-rank pass + word-window tighten (seg_sub_min 0.5): 0.516 → score 0.699**.
      Oracle ceilings: best single segment 0.62, best word window 0.93 → selection is the loss.
- [x] Defaults set in `pipeline.SPAN_STRATEGY`. Re-rank pass costs ~5 s/conversation.
- [x] Re-rank pass returning verbatim words: 0.689 < lexical tightening 0.700 → `rerank_quote: 0`.
- [x] `example.py` now serializes requests (lock) and is budget-aware: skips the re-rank pass
      if < 14 s remain, skips the LLM entirely if < 16 s remain (lexical answers instead).
- [x] HTTP end-to-end on a quiet machine (`local_eval_http_v2.log`, 19:05): **score 0.700**,
      accuracy 0.977, tIoU 0.515, 0 timeouts, round trip mean 26.5 s / worst 45 s of 60 s.
      (First run v1 timed out — that was GPU OOM from YOLO + a duplicated server, not the pipeline.)
- [x] First official validation run (18 Sep ~19:40–19:52): 19/19 conversations answered,
      21.8–44.1 s each, no timeouts (Mac must be free of heavy browser tabs: they push Qwen
      into swap and triple the latency). Score: see scoreboard.
- [ ] Ideas: per-question prompts with KV-cache reuse; larger embedder for candidates;
      ask for line + quote in the re-rank pass; Qwen2.5-14B-4bit if latency allows (unlikely).
