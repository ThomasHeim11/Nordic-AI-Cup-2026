# Task1

# Plan for 25 points — Survival Simulator

Target: highest raw survival score at the final evaluation (3 fixed seeds, averaged).
Validation leader today: 1794 s. We: 619 s (cut by latency, not by the policy). Local policy: ~950 s mean.

## Why we lose today

1. **Latency.** Rule: total waiting on our server ≤ 600 s over up to 30 000 ticks → ≤ 20 ms/tick.
   Mac + tunnel = 97 ms/tick → game ended at tick 6190. No policy can score above the latency cap.
2. **Policy.** Runs end by boom–bust starvation at 600–1300 s, while the endgame (t > 1000: ~3 trees,
   ~30+ predators, forced re-breeding every ~100 s) is never reached in good shape.

## What we do

### A. Hosting — measured 18 Sep, the binding constraint

| path                                          | ms/tick (from the 600 s rule) | validation  |
| --------------------------------------------- | ----------------------------- | ----------- |
| Mac + Cloudflare tunnel                       | 97                            | 619 s (cut) |
| cluster n009 + Cloudflare tunnel              | 60                            | 992 s (cut) |
| Mac direct, router port-forward (UPnP), Wi-Fi | 80                            | 750 s (cut) |

Our side is not the cost: policy 0.6 ms/tick, full realistic HTTP tick 3 ms on localhost.
The 60–80 ms is network round trips (Helsinki ↔ Oslo ≈ 20–25 ms RTT; likely a new TCP connection
per tick on the evaluator side, plus Wi-Fi jitter). Options left, in order:

1. Measure connection reuse from n009 (script given). If reconnect-per-tick: nothing on our side
   reduces round trips except proximity to Helsinki.
2. Ethernet cable Mac ↔ router (removes Wi-Fi jitter, maybe −10 ms).
3. Simula admins forwarding a port to n009 (tunnel-free, still Oslo → ~35 ms).
4. A host near Helsinki/Stockholm — every provider considered so far is excluded by the team.
   Realistic ceiling without 4: ~1000–1500 s → top-8 on today's board, not 25.

**Decision (18 Sep, late):** build the policy first (B1 → B2), fix hosting afterwards with the
levers we control (small population = small requests, delayed-ACK off, Ethernet, then re-measure).
A small elite population is good for _both_ survival and latency, so B1 and hosting pull the same way.

### B. Better policy — Saturday

**B1. Engineer the endgame (heuristic, our fallback and the RL teacher)**

- Population target follows food: grow to ~30 while trees > 20, shrink to 3–6 when trees < 8.
- One replacement at a time: an elder breeds once when a child can be fed; no bursts.
- Ripening-aware harvest: fruit reaches 60 energy after 20 s; don't eat it fresh unless starving.
- Elite genes: breed only from top `max_energy` / `hearing_radius` animals (hearing 100 > predator charge range 90).
- Late-game idling: stand still (1 energy/s), face the nearest predator, sprint only on a charge.
- Measure every change on 24 fixed maps on the cluster (`bench_many.py`); keep only proven gains.

**B2. Actor-critic RL (own PPO in PyTorch, equations from the RL lecture)**

- Env: our simulator, 60 parallel processes on n009; every animal = one row through one shared network.
- Obs (~60 features): own energy/age/traits/biome/time, nearest fruits/trees/mates/predators
  (distance, angle, facing), wall vector, population, food per capita.
- Actions: move distance, move direction, turn (Gaussian) + breed (Bernoulli).
- Reward: +1 per tick species alive (the score) + fruit energy − energy lost to predators; γ = 0.999, GAE.
- Warm start: behaviour-clone the B1 heuristic first → net starts at heuristic level, PPO improves from there.
- Checkpoint hourly; score each on 24 maps; ship the best of {heuristic, RL}.

### C. Tuning — continuous

- Cluster evolver / CMA-ES on the final behaviour, 8–16 maps per candidate, optimise the _mean_.

## Also in the plan (drone, from Discord): estimate camera motion online from consecutive frames

instead of the Helsinki-fitted homography — the evaluation flight direction may differ.

## Timeline

| When                        | Owner   | Step                                                                                |
| --------------------------- | ------- | ----------------------------------------------------------------------------------- |
| Fri night                   | done    | cluster + home hosting measured (60 / 80 ms per tick)                               |
| Fri 23:50                   | done    | B1 endgame rewrite: 1002 mean on 24 maps (was 950), variance down                   |
| Fri 23:50                   | done    | B2 code complete and smoke-tested (features, actor-critic, BC, PPO, cluster runner) |
| Sat midday                  | Claude  | RL groundwork: env wrapper, features, BC data + training on A40; go/no-go           |
| Sat afternoon → Sun morning | cluster | PPO, hourly checkpoints scored on 24 maps                                           |
| Sat afternoon               | both    | hosting: small-population run, delayed-ACK off, Ethernet if available; re-validate  |
| Sun 09–12                   | both    | pick best policy, validate on the real scoreboard, freeze                           |
| Sun ~14                     | you     | final evaluation (one attempt)                                                      |

## Definition of done

- Validation ≥ 1800 s from the final host, repeated twice.
- Mean over 24 local maps ≥ 1900 s.
- Training code + model saved in the repo (top-5 must hand it in by Sun 20:00).

# Task2

## Plan for max points — Drone Flyby

Target: mAP@0.50 on an unseen 250-frame sequence. Validation leader 0.830 · 5th 0.749 · 10th 0.485. We: 0.005.

### What we have

- Pipeline: YOLO detector on the 960×540 view → boxes lifted to the 4K frame → **tracker** that
  dead-reckons every object with a per-frame homography (objects stay reported ~20 frames after one
  sighting) → **camera policy** (L0 → four L1 tiles → L1 sweep of the entry band + L2 dives on small /
  uncertain tracks), every command checked against the rules, true camera state tracked.
- Local score on the Helsinki scene: 0.25 (baseline) → **0.90**. 58 ms/frame on the Mac.
- Recording of every validation frame (`recordings/`, 182 frames of the unseen terrain).

### Why validation gave 0.005

1. **Detector did not generalize** — trained on 25 frames of one orthophoto scene; the validation
   scene is Google-Earth-style 3D texture. It fired ~2000 `ta-ta` on bushes/cars and missed a hangar
   and five aircraft in plain view.
2. **28 % of frames skipped** — 1.4 MB per frame through the home tunnel exceeded the 333 ms interval.
3. **Camera commands refused** — request views can be stale; fixed (true camera state kept).
   Organizers: evaluation flight direction and object appearance "may differ" from validation.

### What we do (ordered by expected gain)

1. **Detector that generalizes** (the big one)
   - Copy-paste synthetic data: Helsinki object cut-outs pasted at correct scale onto the recorded
     validation terrain + Helsinki backgrounds, empty crops as negatives (`make_synth.py`, 4000 imgs).
     Training yolo11 s / m / l on the A40 — running overnight (`cluster_train.sh`).
   - Pick by `check_recordings.py` (what each model sees on the unseen frames: a healthy model finds
     a handful of confident objects per frame, not hundreds of one class) + Helsinki val.
   - Round 2: **pseudo-label** the recorded validation frames with the best model (high-confidence
     boxes only) and retrain — real terrain, real object renderings.
   - Heavier augmentation (colour/blur/JPEG/scale), test-time flip augmentation if latency allows.
2. **Serve from the cluster** (`drone_serve.sbatch` / `cluster_serve.sh`): university bandwidth ends the
   frame skipping, and the A40 makes the large model affordable (~30 ms per frame vs 333 budget).
3. **Online motion estimation**: register consecutive same-level views (feature matching / phase
   correlation) to estimate the homography per sequence instead of assuming Helsinki's direction;
   fall back to the fitted matrix. Required — the evaluation flight direction may differ.
4. **Output hygiene**: per-class confidence thresholds learned from the recordings (kill the `ta-ta`
   flood), class-agnostic duplicate suppression, cap detections per frame.
5. **Validate after every model** (unlimited) and keep recording — every validation run adds 250 more
   unseen-terrain frames for pseudo-labelling.

### Timeline

| When          | Step                                                                           |
| ------------- | ------------------------------------------------------------------------------ |
| Sat morning   | read s/m/l results → `check_recordings` → pick → serve from cluster → validate |
| Sat midday    | pseudo-label recordings → retrain → validate; per-class thresholds             |
| Sat afternoon | online motion estimation; re-validate                                          |
| Sat evening   | freeze the model; two clean validation runs from the final host                |
| Sun ~14       | final evaluation (one attempt)                                                 |

### Definition of done

- Validation ≥ 0.75 (top-5); stretch ≥ 0.85 (1st).
- `check_recordings` shows sensible per-class counts; 0 refused camera commands; 0 skipped frames.

# Task3

## Plan for max points — Medical Appointment

Target: 0.4·accuracy + 0.6·evidence tIoU on 38 unseen conversations.
Validation board: 1.000 and 0.992 (two teams; the validation audio is not in our training set, so
these are earned and prove near-perfect evidence is achievable) · 3rd 0.830 · 5th 0.800 · 10th 0.780.
We: **0.698** (offline 0.706). Every point from here is in the evidence timestamps.

### What we have
- Mac pipeline: whisper-large-v3-turbo (word timestamps) → Qwen2.5-7B answers all 10 questions in
  one prompt and quotes its evidence → quote located on word timestamps → second LLM pass re-ranks
  candidate lines → span tightened to the fact-bearing words → silence-based boundary snapping.
  Budget-aware server (drops passes when time runs out); 22–44 s per conversation online, 0 timeouts.
- Accuracy **0.977** (answering is solved). Evidence tIoU **0.52** (the whole gap).
- What we learned from the 195 gold spans: a gold span is one or more whole *utterances*; it starts
  at speech onset after a silence (±0.05 s) and ends at the last word. Whisper's word starts are
  0.2–0.4 s early (snapping fixes that). Our utterance *choice* is right 68 % of the time, wrong
  passage 19 %, poor overlap 10 %. A perfect choice with today's boundaries → tIoU ≈ 0.74 →
  score ≈ 0.83; perfect choice + perfect boundaries → ≈ 0.98.
- `backend_torch.py` + `MEDICAL_BACKEND=torch`: the identical pipeline on CUDA (HF transformers),
  smoke-tested; `cluster_medical.sh eval|serve`; the medical bundle is already on the cluster.

### What the cluster (A40, 46 GB) lets us do that the Mac cannot
1. **Bigger selector model** — Qwen2.5-14B-Instruct in bf16 (28 GB) fits; 32B in 4-bit (AWQ/GPTQ,
   ~19 GB) if the quantised kernels install on ARM. The answer pass is already 0.977; the gain is in
   *which utterance* it cites and in the re-rank pass. Expected: selection 68 % → 80–85 %.
2. **Choice set = utterance units** — cut the transcript into silence-delimited utterances (we detect
   them already) and have the LLM pick unit ids (1–3 consecutive units) instead of free quotes. The
   answer space then matches the annotators' convention exactly; boundaries come from the units.
3. **A learned re-ranker on our own gold** — we have 195 (question → gold utterance) pairs plus every
   other utterance as a negative. Fine-tune a small cross-encoder (e.g. a BERT/DeBERTa-base scorer,
   minutes on the A40) to score (question, utterance) and combine it with the LLM's choice. This is
   the most likely explanation for the two ~1.0 teams: a model trained on the annotation convention.
   Cross-validate over the 39 conversations (leave-conversations-out) to get an honest number.
4. **Better/denser ASR** — whisper-large-v3 (non-turbo) or WhisperX forced alignment for tighter word
   times; on the A40 either runs in seconds, so it costs no budget.
5. **Self-consistency** — run the selection twice (different prompt orderings / temperature 0 vs
   sampled) and keep agreeing spans; when they disagree, fall back to the re-ranker's choice. The A40
   has ~40 s of headroom per conversation for this.
6. **Serve from the cluster** — no swap, no browser-memory risk, ~10–15 s per conversation; Mac
   pipeline (7B) stays the fallback host.

### Order of work (Saturday)
| When | Step | Success check (offline, 39 conversations, cached ASR) |
|---|---|---|
| morning | `cluster_medical.sh eval` with 14B → compare with 0.706 | +0.03 or more → keep |
| morning | utterance-unit choice prompt (ids, 1–3 units) | tIoU up, accuracy ≥ 0.97 |
| midday | cross-encoder re-ranker trained on gold, leave-conversations-out CV | selection ≥ 80 % |
| afternoon | combine LLM + re-ranker; self-consistency if time | offline ≥ 0.80 |
| afternoon | serve from the cluster; validation run (unlimited, keep the audio? no — not provided) | validation ≥ 0.80, worst case < 45 s |
| evening | freeze; two clean validation runs | stable |
| Sun ~14 | final evaluation | — |

### Definition of done
- Offline ≥ 0.80 on the 39 training conversations (leave-out CV for anything trained on them).
- Validation ≥ 0.80 (top-3), two consecutive runs, worst conversation < 45 s, 0 timeouts.
- Mac fallback (7B) still verified and reachable.
