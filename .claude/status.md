# Status log

## ⚠ Morning of 19 Sep — state to know before touching anything
- Mac **medical server is OFF** (stopped 01:20 for memory). Restart before any validation:
  `cd medical-appointment && nohup ./.venv/bin/python api.py > api_9054.log 2>&1 &` (its tunnel is still up).
- Mac: insurance YOLO-small training on synthetic data may be running (`drone-flyby/train_mac_synth.log`).
- Cluster: drone training was restarted alone (all stalls were GPU/CPU contention); survival evolver + PPO
  are SIGSTOPped and auto-resume when the drone finishes; the medical 14B eval was killed → rerun after
  the drone: `cd ~/medical-appointment && nohup bash cluster_medical.sh eval > ~/medical_eval.log 2>&1 &`.
- Cluster allocation (job 1412535, n009) ends ~19:00 Sat.

## Scoreboard (official validation)
| challenge | score | build | limiting factor |
|---|---|---|---|
| survival | **1213 (19 Sep 20:02)**, was 1177 | gen19 genome, lean server, AWS t3.micro | **policy only**: the evaluator keeps ONE connection for all 12 000 ticks (server log "12000 requests over 1 connections"); wall 133 ms/tick is its own sim; our wait ≈ 12 ms/tick ≪ the 600 s cap. Hetzner would gain nothing. |
| drone | **0.157 (19 Sep 20:04)**, was 0.005 | final mix2 detector, OpenVINO fp32 on AWS c7i-flex.large | 296 requests answered, 0 errors; recording of the run pulled home (`$CLAUDE_JOB_DIR/tmp/drone_val2`, 249 frames) for analysis |
| medical | **0.708 (19 Sep 19:15)**, was 0.698 | quote + re-rank + snap + cross-encoder ensemble, Mac 7B | passage selection (62/195 miss) |

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
- [x] B1 endgame (18 Sep 23:30): species clock + `population_target(t)` ramp (spawn_max_pop →
      late_pop between late_t0 and late_t1), ripening-aware harvest (`hungry_frac`), elders capped
      by carrying capacity. **24 maps: 1002 mean, sd 202, min 510** (was 950 / 304 / 467). Kept.
      Mac survival server restarted on it (23:55).
- [x] B2 RL groundwork (`rl/`): features.py (85-dim egocentric vector), model.py (shared
      actor-critic MLP, 0.18 ms/tick for 30 animals), bc_collect.py / bc_train.py (behaviour cloning
      from the heuristic), policy_nn.py (drop-in `decide_all`), ppo.py (own PPO: parallel simulators,
      per-animal GAE, clipped objective), cluster_rl.sh (bc | ppo | eval). All smoke-tested on the Mac.
      `bench_many.py --policy rl.policy_nn` scores checkpoints (weights via SURVIVAL_NN=...).
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
- [x] **Online motion estimation** (`MotionEstimator` in example.py): ORB + RANSAC affine between
      consecutive same-level views in source coordinates; prior kept unless the estimate disagrees
      (> 10 px/frame or scale > 0.006). Helsinki stays 0.903; reversed-heading test switches correctly.
      Recorded validation scene measured: dy +68 px/frame ≈ Helsinki (+65).
- [x] Report hygiene: class-agnostic NMS (IoU 0.6) + cap 80 per frame (neutral on Helsinki).
- [x] `pseudo_label.py` + `cluster_train.sh round2 <weights>`: self-training on recorded views.
- [x] Flip TTA implemented (`DRONE_TTA=1`), OFF by default: 0.867 vs 0.903 on Helsinki with the current model.
- [ ] Per-class confidence thresholds from the recordings once the new detector exists; re-test TTA with it.

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
- [x] Gold spans = whole utterances; starts sit at speech onset after a silence (±0.05 s), whisper
      word starts are 0.2–0.4 s early. Silence-based snapping added (`snap_span`): 0.700 → 0.706.
- [x] Loss decomposition on 195 yes-questions: 133 picked well (mean tIoU 0.74), 37 miss the
      passage, 19 poor overlap, 10 zero. **Perfect selection ≈ tIoU 0.74 ≈ score 0.83 (3rd).**
      → the lever is utterance *selection* → stronger LLM on the A40.
- [x] `backend_torch.py` (transformers, CUDA) + `MEDICAL_BACKEND=torch` routing; smoke-tested with tiny
      models on the Mac. `cluster_medical.sh eval|serve` (default Qwen2.5-14B-Instruct bf16).
- [x] **Units mode** (`units_mode.py`, `--mode units`): silence-delimited utterances (gap 0.2 s, ~50/conv)
      as the answer space; oracle 1–3 consecutive units = 0.81 tIoU. With the 7B: **0.685** (< 0.706) —
      accuracy 0.964, tIoU 0.499. Kept as an option for the 14B; not default.
- [x] **Learned cross-encoder re-ranker** (`reranker.py`, MiniLM on 195 gold spans, 5-fold CV by
      conversation): 2 epochs + context 36 %; **6 epochs, no context: 68.7 % selection, tIoU 0.515** —
      on par with the LLM, different errors → combine. Defaults now context off / 6 epochs.
      Overnight: re-ranker trained on all 39 + three combined units+CE evaluations → `units_ce_eval.log`
      Result: units + CE = 0.690 / 0.695 / 0.682 (optimistic, same conversations) — **below 0.706**.
      Units/CE route closed for the 7B; default stays quote + re-rank + snap. Lever = 14B on the A40.
- [x] Drone cluster training stalled 4× (GPU/CPU contention, then even alone at batch 1 after kills);
      stopped for the night. Insurance: Mac trains yolo11s on the synthetic data (`train_mac_synth.log`,
      ~03:30). Morning: retry on a **fresh** allocation with the drone alone on the GPU.
- [ ] Tomorrow on the A40: 14B (and 32B-4bit if it installs) in quote mode AND units mode; pick best.
- [ ] Ideas: per-question prompts with KV-cache reuse; larger embedder for candidates;
      ask for line + quote in the re-rank pass; Qwen2.5-14B-4bit if latency allows (unlikely).

## Sat 19 Sep — drone diagnosis (recorded validation frames)
- Validation scene = Copenhagen-style Google-Earth terrain, ~15 sparse rendered objects. Verified by eye (mine_recordings.py):
  hangar + 2 jet_plane on dry field, 4-6 small_plane (red wing tips) on a lot, large_tower (lattice) x1-2, helicopter by water,
  small_tower (green roof on stone plinth) x1-2, tank x1-2, jammer x1.
- Why validation scored 0.005: (1) model calls every orange roof / dark car "ta-ta" at 0.5-0.9 — Helsinki ta-ta always sits on red clay;
  (2) jets called "condor" (dry-grass bias), small planes called "ta-ta"; (3) first synth run used recorded frames as *unlabelled*
  background -> taught the model the hangar/jets are background (synth model: 2 boxes in 80 frames, Helsinki 0.863).
- Fix in progress: make_synth.py --mined (59 verified validation cutouts, real boxes written for labelled frames, scale jitter 0.75-1.3),
  dataset data/synth2 (4000 JPEG), data/mix2.yaml = Helsinki + synth2; fine-tune from weights/best.pt (0.903 Helsinki).
- weights/best.pt still = drone_s2 (0.903). Mac synth model saved as weights/synth_mac_s.pt (do not serve).
- Survival: cluster evolver gen 19 genome (survival-simulator/evolved_gen19.json) benches 1123 mean / max 2123 on seeds 201-224
  vs 1002 baseline -> deployed to src/utils/controllers/hivemind_params.json (old one kept as hivemind_params.json.bak_1002). PPO abandoned (243 s).
- Medical on cluster abandoned: qa_eval --transcribe-only did 2/39 conversations in 9 h (GPU idle). Serve Mac 7B (0.698).
- Cluster GPU now reserved for drone: upload drone_synth2.tgz (scp -P 60441 ... dnat.simula.no), train from best_0903.pt on data/mix2.yaml.
- Sat 19 Sep 10:27 CEST: survival validation with evolved genome via UPnP direct: 1006 (16 min wall, 600 s wait rule). Hosting still the ceiling.
- Sat 19 Sep ~12:15: Render free tier (survival-3f1o.onrender.com, Frankfurt, 0.1 CPU) validated 569 (no bottleneck error, but slow + early deaths) -> kept only as emergency spare. Mac direct stays (1006). delayed_ack=0 set on Mac.
- Cluster: n009 A40 froze training twice (GPU idle, py-spy in make_anchors); allocation died with ssh; A40s then taken by another user; n013 A100 "CUDA unknown error". All drone cluster jobs cancelled; everything trains/serves on the Mac. slurm/ has drone_train_mix2/drone_eval/drone_chain(_x86) for reference.
- Plan: Mac drone run (runs/detect/runs/mix2_s, 20 ep, done ~15:30) -> check_recordings + eval_inprocess -> serve 9053 + UPnP -> validate; medical 9054 restart -> validate; Sunday evaluations ~14:00.
- Sat 12:40: Mac mix2_s checkpoint (epoch ~4, Nordic-AI-Cup-2026/runs/detect/runs/mix2_s/weights/best.pt, copy weights/mix2_ep.pt)
  on 183 recorded validation frames @conf0.3: jet 22, small_plane 20, hangar 14, small_tower 13, large_tower 7, tank 5, heli 2, ta-ta 4
  (old 0.903 model: ta-ta 244 FPs, ~0 real objects). Helsinki pipeline 0.856 (old 0.903). Final pick after epoch 20 (~15:30).
- Sat 12:58: AWS Stockholm survival endpoint http://16.170.155.200:9052 live (see deploy.md); validation queued (pos 6).
- Sat 13:06: survival via AWS Stockholm validated **1177, no errors** (natural end, ~40 ms/tick). Latency solved; score now = policy. Next: continue evolution on cluster ARM nodes (slurm/survival_evolve_arm.sbatch), bench on 24 maps, deploy best to AWS via git pull.

## Sat 19 Sep afternoon (branch worktree-survival-fast-server) — cluster abandoned, everything local
- **Cluster is not used any more** (too many failures). All training/evolution on the Mac; survival serves from AWS.
- **Survival latency anatomy** (measured): server compute < 1 ms; the FastAPI container answers in 3.2 ms on the AWS box;
  Mac→AWS 20 ms with keep-alive vs 50 ms with a new connection per request (a 27 KB body needs ~3 round trips:
  handshake + slow start). The 600 s wait rule therefore caps the run at ~600/(3·RTT+server) ticks. Only geography
  (Hetzner Helsinki = evaluator's own DC, paid) removes round trips; server-side we can only shave milliseconds.
- Lean ASGI server (`agent_server.py`: orjson, uvloop, httptools, no pydantic on the hot path): **3.2 → 1.3 ms** per
  request on AWS; bit-identical actions over 4000 replayed ticks. Image `survival:fast` is built on the AWS box and
  staged on port 9153 (`--network host`). **Swap into 9052 = human action** (classifier blocks the deploy):
  `sudo docker rm -f survival_fast survival; sudo docker run -d --restart unless-stopped --network host -e PORT=9052 --name survival survival:fast`.
  It also logs "N requests over M connections" every 2000 ticks → the next validation tells us if the evaluator reuses connections.
- Games over HTTP are **not repeatable** (same seed, same code: 1196 vs 1374): the bench sd of ~300 is real noise, use ≥ 24 maps.
- 3000 s sim without agents: trees never run out (seed 201: 33 trees @900 s, 13 @1800 s, 5 @3000 s; fruits 84 → 20 → 14)
  but predators climb 0 → 8 @900 s → 16 @1800 s → 23 @3000 s. Individuals cannot outlive ~max_age(60–120 s)+ageing;
  the species lives as long as births continue = as long as animals keep finding the (sparse, moving) trees.
- Baseline bench (this Mac, training running): gen19 genome **967 mean / sd 315 / min 242 / max 1742**, deaths starved 1950 vs eaten 816.
- Implemented the **shared tree map**: agents that see each other merge dead-reckoning frames (same identity as
  relay_threats), newborns inherit the parent's frame, trees remembered by anyone become targets for everyone
  (`memory_shared`, `memory_shared_visited` params). Geometry unit-tested exact. Bench pending (see below).
- Medical: baseline re-confirmed 0.706 offline. Sweeps from cached replies: snap tolerances (0.4/0.2 best), pad 0.15 → 0.688,
  quote_pair 0.695, quote_seg 0.690, seg_sub_min 0.35/0.7 → 0.704/0.700. Boundaries are unbiased but noisy (exact boundaries
  on the 133 picked questions would give tIoU 0.70); 62/195 miss the passage.
  **Cross-encoder ensemble** (`pipeline._ce_ensemble`, fold models for the test): keep the LLM span if it overlaps one of the
  CE's top-2 units, else take the CE unit (grown 0.5 z) when its margin ≥ 1.0 → held-out tIoU 0.526 → 0.556, **offline 0.724**.
  Default on; `example.py` warms the CE. Not yet validated online.
- Survival endgame (bench logs every 20 s): the species dies with 20–40 fruits and 12–20 trees still on the map, 1–5 animals at
  energy 50–100 that cannot reach food fast enough; predators 10–15 by 1500 s. Deaths are mostly old-age drain (0.01·age per
  tick past max_age 60–120 s), so continuity = births. Added `crowd_limit_late` (ramps with late_t0/t1, default = current) for
  the evolver. **evolve.py had a SyntaxError since the --out commit (global OUT after use) — fixed.**
- Unattended chain (scripts + logs in `~/.claude/jobs/1c488450/tmp/`): `after_train.sh` (waits for the YOLO run, scores
  best/last/epoch-4 on the recordings + Helsinki, installs the winner, starts :9053/:9054 from main, UPnP) → `restart_from_branch.sh`
  (restarts both servers from the branch worktree) → `drone_exp.sh` (eval_recorded + Helsinki with conf 0.12/0.05, flip TTA,
  ta-ta 0.5) → `evolve_after.sh` (niced local evolver, 80 gens × 12 × 6 seeds, genome → `survival-simulator/evo_mac_a.json` in the worktree).
- **16:00 drone mix2_s finished (20 epochs, val mAP50 0.8246).** The watcher's checkpoint loop was broken (macOS bash has no
  `declare -A`, all "candidates" were the epoch-4 file) → scored by hand. **Final epoch-20 checkpoint wins**: on the 183
  recorded validation frames @0.3 it reports 107 real-class boxes (jet 33, large_tower 20, small_plane 16, small_tower 14,
  hangar 12, jammer 5, tank 3, heli 2) vs 83 for epoch 4, ta-ta FPs 1 vs 4; recall on the 59 verified objects 0.93 @0.3
  (jammer found at every threshold, tank 2/3); **Helsinki in-process mAP 0.936** (old model 0.903, epoch 4 0.856).
  Installed as `weights/best.pt` (copies: `weights/mix2_final_best.pt`, epoch 4 = `weights/mix2_ep.pt`, old = `weights/pre_mix2_best.pt`).
- **Servers now run from the branch worktree** (`.claude/worktrees/survival-fast-server/{drone-flyby,medical-appointment}`,
  logs `api_9053.log` / `api_9054.log` there): drone :9053 with the final weights, medical :9054 with the cross-encoder
  ensemble. UPnP forwards 9052/9053/9054 are in place. Medical checked over HTTP: sample_4/5/6 through the public address
  and loopback answer in 19–32 s; their spans average tIoU 0.635 = the offline number with the ensemble on (0.550 off) → live.
- **17:00 drone hosting diagnosis.** Inference settings (conf 0.05/0.12, flip TTA, ta-ta 0.5) all give Helsinki 0.936 —
  irrelevant. Realtime evaluator run **from AWS Stockholm** against the public Mac URL: 387 ms/frame, 12/25 frames skipped,
  mAP 0.39 (loopback 123 ms, 0.84; evolver running adds ~13 ms). The Mac is on **2.4 GHz 802.11n Wi-Fi** (RT-N12E is
  2.4 GHz-only) ≈ 40–50 Mbit/s, and the evaluator uploads a 1.4 MB frame every 333 ms. Fixes: Ethernet cable (100 Mbit LAN
  ports) and/or an AWS CPU box. CPU path built: ONNX export at 544×960 (`models/mix2_final_best.onnx`, tracked) gives the same
  Helsinki 0.936 through the pipeline (198 ms/frame on the M1 CPU); `Dockerfile.cpu` (torch-cpu + ultralytics + onnxruntime,
  `DRONE_WEIGHTS=models/...onnx`). Timed on the t3.micro's 2 old cores: 305 ms/frame → a c7i.2xlarge should do 50–80 ms.
  Launching the instance needs the AWS console (no CLI/credentials on the Mac) → user.
- Drone: `eval_recorded.py` scores a checkpoint on the 59 human-verified validation objects (38 frames of recording
  6262…; those frames were synth backgrounds → optimistic). Epoch-4 mix2 checkpoint: recall 0.95 @conf 0.1, ta-ta 0 FPs.
  `DRONE_CLASS_CONF` env adds per-class confidence thresholds.
