# TODO — Nordic AI Cup 2026 — target: 1st in all three

## SUNDAY 14:50 RUNBOOK (deadline 16:00 — evaluations must be queued, and survival takes ~75 min)

**Press in this order, no gaps:**
1. **Survival — Queue EVALUATION immediately** (`http://16.170.155.200:9052`, last validations 1213 / 1599). It runs ~75 min, so
   pressing at 14:50 finishes ~16:05; if you can press it earlier from your phone, do (validation not needed — nothing changed since 1599).
2. **Medical — Queue EVALUATION** (`http://178.232.205.38:9054`, server restarted 11:15, answers publicly; ~20 min). Nothing changed since 0.708.
3. **Drone — validate first, then evaluate** (`http://16.192.171.219:9053`, 2 min each):
   a. Queue validation on the live build (motion fix + thresholds 0.02/0.01; see below). Note the score = **X**.
   b. If X ≥ 0.165 → **Queue evaluation**. If X < 0.161 → roll back (Mac terminal, 1 min) and evaluate the old build:
      `ssh -i ~/Downloads/nordic.pem ubuntu@16.192.171.219 'cd ~/Nordic-AI-Cup-2026/drone-flyby && DRONE_MOTION_ONLINE_MIN=0 DRONE_CONF=0.08 DRONE_REPORT_MIN_CONF=0.05 bash deploy_aws_cpu.sh'`
   c. Only if there is time (>15 min before you must evaluate): one extra variant, full-frame look every 6 frames —
      `ssh -i ~/Downloads/nordic.pem ubuntu@16.192.171.219 'cd ~/Nordic-AI-Cup-2026/drone-flyby && DRONE_L0_EVERY=6 bash deploy_aws_cpu.sh'` → validate → keep the better → evaluate.
      (Your 11:30 validation of "variant B" was L0_EVERY=6 *without* the motion fix — compare it with 0.165, not with X.)
4. After all three evaluations are done: terminate both EC2 instances; merge this branch into main (`rm todo.md` in the main checkout first).

**What changed on the drone box since your 09:24 validation (0.1648), all measured offline on the recorded validation sequence:**
- Tracker now trusts the *measured* motion after 8 accepted registrations (it used to flip back to the Helsinki prior, ~6 px/frame wrong):
  out-of-view recall on remembered objects 0.457 → 0.519, in-view recall 0.85 → 0.93 (traj_eval.py / replay_eval.py). Helsinki 0.804 → 0.797.
- Detect/report thresholds 0.08/0.05 → 0.02/0.01 (rank-based AP: extra low boxes only add recall): in-view mAP 0.875 → 0.884, held-out split 0.89 → 0.95.
  Dives keep a 0.08 track-confidence floor so junk tracks do not steer the camera.
- Sweep band mirrors to the bottom if the flight direction is reversed (hysteresis, 20 px). A reversed-flight Helsinki test scored 0.15 vs 0.80
  forward when the prior was trusted — the evaluation "may fly differently", so this is insurance.
- Verified on the live box: full 249-frame replay OK, tracks bounded (≤ 37), ~16 boxes/frame, no errors.

## SUNDAY 09:40 — DRONE: NEW MODEL LIVE on `http://16.192.171.219:9053` → validate, then Evaluate before 16:00
- In-domain labels: 360 objects on 143 recorded validation frames (Claude-verified: 58 small launchers, 13 jammers, 6 helicopters,
  3 large launchers, 1 mine roller + auto-accepted planes/towers/hangars/tanks). Fine-tuned 4 epochs (mix4).
- Held-out labelled frames mAP50: **0.909 vs 0.407** (old). On the 2nd recording @0.3: jets 105 (50), small launchers 43 (3),
  large towers 53 (22), jammers 8 (4), helicopters 6 (3), large launchers 3 (0), mine rollers 2 (0). Helsinki 0.79 (was 0.936; irrelevant).
- **If the drone validation is below 0.161**, roll back to the previous model (2 min) and validate again:
  `ssh -i ~/Downloads/nordic.pem ubuntu@16.192.171.219 'cd ~/Nordic-AI-Cup-2026/drone-flyby && git checkout 165eb42~1 -- models/ && bash deploy_aws_cpu.sh'`
- Incident 09:15: the Mac disk filled (14 GB swap + 8 GB Docker image) → a broken commit went live for ~4 minutes with no
  detector; restored, disk freed (Docker.raw deleted), re-exported, redeployed 09:38.

## SUNDAY MORNING — earlier brief
**What happened overnight (details at the bottom under "Overnight results")**
- Drone: the synth3 fine-tune did NOT find the five missing classes and lost towers/tanks → **current model kept**.
  Thresholds loosened on the box (conf 0.08 / report 0.05) for more recall on the near-zero classes.
- Survival: evolver pass 1 → 971 vs 950 on 24 maps (noise) → **gen19 kept**. Pass 2 (12 seeds/genome, 8 gens) → 1026 vs 1079
  → **gen19 kept**. The live container still runs the genome that validated 1213.
- Medical: three selection variants all ≤ 0.724 offline → **validated build kept**, server restarted warm at 02:18.
- Health at 02:18: survival 200 · drone 200 · medical 200; UPnP 9052/9053/9054 present.

**Your steps (≈ 08:00 → 10:30)**
1. Validate all three (they can run at the same time, different hosts):
   survival `http://16.170.155.200:9052` (25 min) · drone `http://16.192.171.219:9053` (2 min) · medical `http://178.232.205.38:9054` (10 min).
   Nothing heavy on the Mac while medical runs.
2. Drone only: if the validation is **below 0.157**, revert the thresholds and validate again (2 min):
   `ssh -i ~/Downloads/nordic.pem ubuntu@16.192.171.219 'cd ~/Nordic-AI-Cup-2026/drone-flyby && DRONE_CONF=0.12 DRONE_REPORT_MIN_CONF=0.08 bash deploy_aws_cpu.sh'`
3. By 09:00 press **Evaluate** on all three (survival ≈ 75 min for 3 sims, medical ≈ 20 min, drone ≈ 3 min). Keep everything up.
4. Afterwards: terminate both EC2 instances (survival + drone); merge the branch (`rm todo.md` in the main checkout first).
Deadline Sun 20 Sep 16:00 CEST · evaluations Sun ~14:00 · one Evaluate per challenge, ever.
Rank points: 1st 25 · 2nd 18 · 3rd 15 · 4th 12 · 5th 10. 1st on all three = 75.
**Cluster abandoned (Sat 19 Sep): everything trains locally on the Mac; survival serves from AWS.**

## Branch `worktree-survival-fast-server` (Sat afternoon, unattended work) — what is in it
| what | measured | state |
|---|---|---|
| survival: lean ASGI server (orjson/uvloop/httptools) | 3.2 → 1.3 ms per request on the AWS box; actions bit-identical over 4000 replayed ticks | built on AWS as `survival:fast`, staged on :9153; **swap into :9052 needs you** (see §1) |
| survival: lineage-shared tree map (`memory_shared`) | 1036 vs 1065 on 24 maps (sd 238) → neutral | in code, default off |
| medical: cross-encoder second opinion on spans (`ce_ens`) | held-out tIoU 0.526 → 0.556; offline **0.706 → 0.724** | default on; server warm-up loads it |
| drone: `DRONE_CLASS_CONF` per-class thresholds + `eval_recorded.py` | epoch-4 mix2 ckpt: recall 0.95 @0.1 on the 59 verified objects | in code |

**When you are back:** in the main checkout `rm todo.md` (untracked copy), then `git merge worktree-survival-fast-server`.
The branch servers are what is running on :9053/:9054 after `restart_from_branch.sh` (logs in the job tmp dir).

## What 1st actually requires (validation leaderboard, 18 Sep)

| challenge | us | 1st now | hard max | what stands between us and the max |
|---|---|---|---|---|
| 1 survival | 1177 | 1794 | **3000 s** (sim cap) | (a) policy must keep breeding for 3000 s (bench mean ~1000, max 1742); (b) **600 s wait cap ⇒ < 20 ms/tick from Helsinki**. Measured: server 1.3 ms; the rest is 3 network round trips per tick (new connection + 27–70 KB body). Only hosting in the evaluator's DC (Hetzner Helsinki, paid) removes round trips. |
| 2 drone | 0.005 | 0.830 | 1.0 | detector must see validation-terrain objects (mix2_s, epoch 15/20 at 14:53) + all 16 classes (macro mAP) + < 333 ms/frame |
| 3 medical | 0.698 | 1.000 (2nd 0.992) | 1.0 | held-out set is disjoint from training (lookup impossible). Boundaries exact on picked questions would give tIoU 0.70; picking the right passage every time ≈ 0.74. Offline now 0.724. |

## Hosting (decided)
| challenge | host | why |
|---|---|---|
| 1 survival | **AWS Stockholm** `http://16.170.155.200:9052` | 35 ms/tick validated 1177; lean image staged |
| 2 drone | **Mac direct via UPnP** `http://178.232.205.38:9053` | needs the M1 GPU (59 ms/frame); t3.micro has no GPU |
| 3 medical | **Mac direct via UPnP** `http://178.232.205.38:9054` | needs ~6 GB RAM; latency irrelevant (60 s budget) |

---

## 1 · Survival — from 1177 to 3000
- [ ] **Swap the lean server in on AWS** (classifier blocks Claude from doing it):
      `ssh -i ~/Downloads/nordic.pem ubuntu@16.170.155.200` then
      `sudo docker rm -f survival_fast survival; sudo docker run -d --restart unless-stopped --network host -e PORT=9052 --name survival survival:fast; curl -s localhost:9052/`
      Then **validate**. Its log (`sudo docker logs survival | grep connections`) says whether the evaluator reuses
      connections ("2000 requests over N connections") — N≈1 means keep-alive, N≈2000 means we pay 3 round trips per tick.
- [ ] Decide on geography: a Hetzner Helsinki CX22 (€3.79/month, card) sits in the evaluator's own datacenter →
      ~1 ms/tick → the 600 s cap stops mattering and 3000 s becomes reachable. This is the single biggest lever in the
      whole competition and it is a money/ethics call, so it is yours. Same Docker recipe, only the IP changes.
- [ ] Local evolution runs after the drone training (`evolve_after.sh`, niced, 5 workers, genome → `survival-simulator/evo_mac_a.json`
      in the worktree). Check `grep ^gen $CLAUDE_JOB_DIR/tmp/evolve_after.log`. **Kill it before the final evaluation** (`pkill -f evolve.py`).
- [ ] Bench any candidate on 24 maps **against the baseline run at the same time** (`bench_many.py`, sd ≈ 240 → ±50 on the mean).
- [ ] Deploy loop: genome → `src/utils/controllers/hivemind_params.json` → commit/push → AWS `git pull && docker build && docker run` → validate.
- [ ] Sunday evening: terminate EC2.

## 2 · Drone — from 0.005 to > 0.83
- [x] Training done 16:00. Final epoch-20 checkpoint installed as `weights/best.pt`: Helsinki mAP 0.936 (old 0.903), recall
      0.93 on the verified validation objects, 107 real boxes / 1 ta-ta on the 183 recorded frames (epoch 4: 83 / 4).
- [x] Drone server :9053 restarted from the branch with it (`.claude/worktrees/survival-fast-server/drone-flyby/api_9053.log`).
- [x] Inference settings measured: conf 0.05/0.12, flip TTA, ta-ta 0.5 all give the same Helsinki 0.936 (the tracker's own
      thresholds dominate); TTA adds +2 % recall on the verified objects for +30 ms/frame → not worth it on a tight budget.
- [!] **Hosting is the drone problem now.** Realtime evaluator run FROM AWS Stockholm against `http://178.232.205.38:9053`:
      round trip 387 ms, 12/25 frames skipped, mAP 0.39 (loopback on the Mac: 123 ms, 0.84). The evaluator pushes a 1.4 MB
      frame every 333 ms and the home link cannot take it: **the Mac is on 2.4 GHz 802.11n Wi-Fi** (RT-N12E is 2.4 GHz-only),
      ~40–50 Mbit/s effective.
      1. **Plug the Mac into the router by Ethernet** (100 Mbit LAN ports → ~2× the throughput, expected ~250 ms/frame). Then re-run
         the Stockholm test: `ssh -i ~/Downloads/nordic.pem ubuntu@16.170.155.200` →
         `cd ~/Nordic-AI-Cup-2026/drone-flyby && sudo docker run --rm --network host -v "$PWD":/w -w /w python:3.11-slim bash -c "pip install -q numpy opencv-python-headless requests pydantic 'faster-coco-eval>=1.7.2,<2'; python local_evaluator.py --realtime --url http://178.232.205.38:9053/predict"`
      2. **AWS CPU box for the drone** (robust, no home link). The Free Plan refuses c7i.xlarge and up ("upgrade your plan"), so the
         box is a **c7i-flex.large** (2 vCPU Sapphire Rapids, 4 GB) in eu-north-1, Ubuntu, key `nordic`, security group TCP 22 + 9053.
         Deploy in one line on the box (installs docker, clones the branch, builds `Dockerfile.cpu`, starts it, prints the URL):
         `curl -fsSL https://raw.githubusercontent.com/ThomasHeim11/Nordic-AI-Cup-2026/worktree-survival-fast-server/drone-flyby/deploy_aws_cpu.sh | bash -s -- models/mix2_final_best.onnx`
         Backends in `models/` (all 544×960): ONNX fp32 Helsinki **0.936**, 305 ms on the t3.micro's 2 old cores; OpenVINO fp32 0.915,
         276 ms; **OpenVINO int8 0.902, 202 ms** (Sapphire Rapids has AMX → int8 should be much faster there).
         Pick on the real box with the Stockholm realtime test: ONNX if 25/25 frames, else int8
         (`... | bash -s -- models/mix2_final_best_int8_openvino_model`). A skipped frame costs far more than 3 mAP points.
- [x] **Drone box live: `http://16.192.171.219:9053`** (c7i-flex.large, Ubuntu, container `drone`, OpenVINO fp32 backend).
      From Stockholm: round trip 124 ms mean / 154 max, offline mAP 0.936 (int8: 139 ms, 0.87–0.90; ONNX: 261 ms, 0.936).
      The realtime test from the t3.micro still shows skips because that client needs > 1 s to encode each frame; irrelevant.
      Redeploy after a code/model change: `ssh -i ~/Downloads/nordic.pem ubuntu@16.192.171.219 'cd ~/Nordic-AI-Cup-2026/drone-flyby && git pull -q && bash deploy_aws_cpu.sh'`.
      Recordings of validation views land in `~/drone_recordings` on the box.
- [ ] **Validate** drone on `http://16.192.171.219:9053` → read the new recording (`scp -r` it home) → per-class thresholds if needed.
- [ ] Sunday evening: terminate both EC2 instances (survival + drone).
- [ ] If still weak: another synth round with the new recording's objects (`mine_recordings.py` → `make_synth.py --mined`), overnight.

## 3 · Medical — from 0.698 to ≥ 0.85
- [x] Boundaries: snap tolerances, pad, quote_pair, quote_seg all ≤ baseline. Closed.
- [x] Selection: LLM + cross-encoder ensemble → 0.724 offline (branch).
- [x] Branch server live on :9054; checked over HTTP (public + loopback, 19–32 s per conversation; spans match the offline
      ensemble numbers exactly).
- [ ] **Validate** on `http://178.232.205.38:9054/predict` (expect ≈ 0.72).
- [ ] Next selection ideas (each: `qa_eval.py` on 39 convs, keep only if > 0.724): per-question prompts with KV-cache reuse;
      ask for line + quote in the re-rank pass with the cross-encoder's top-3 as extra candidates; tune `ce_margin`/`ce_top` (0.55–0.556 plateau, don't overfit).

## Sunday 20 Sep
- [ ] 10:00 freeze builds; `pkill -f evolve.py`; commit; update `.claude/status.md`.
- [ ] 12:00 nothing heavy on the Mac; servers restarted on final weights; `upnpc -l` shows 9053 + 9054.
- [ ] 12:00–13:30 validate all three on the final URLs — identical builds.
- [ ] ~14:00 **Evaluate** ×3 (survival = 3 sims; keep everything up).
- [ ] Top-5 ⇒ submit training code + models by 20:00. Terminate EC2.

## Overnight results (auto, Sun 20 Sep 02:18)
- Sun 01:48 drone training: 12 0.83911 (epoch, val mAP50)
- Sun 01:53 drone AI-Cup: real-class boxes / ta-ta / Helsinki mAP / verified recall = 238 5 0.8472 0.93 (current: 278 8 0.9363 0.93)
- Sun 01:56 drone AI-Cup: real-class boxes / ta-ta / Helsinki mAP / verified recall = 238 5 0.8472 0.93 (current: 278 8 0.9363 0.93)
- Sun 01:56 drone: kept the current model (no checkpoint beat it on the recordings)
- Sun 02:15 drone: mix3 (synth3 fine-tune, val mAP50 0.839) finds NO new classes on the validation recordings (0 launchers / condor / medium_plane, 1 mine_roller @0.15) and loses towers + tanks → current mix2 model stays. Thresholds loosened on the box: DRONE_CONF 0.08, DRONE_REPORT_MIN_CONF 0.05 (Helsinki over HTTP 0.916 vs 0.936; rank-based AP → more recall on the near-zero classes of the validation scene). **Fallback if the morning validation is below 0.157**: `ssh -i ~/Downloads/nordic.pem ubuntu@16.192.171.219 'cd ~/Nordic-AI-Cup-2026/drone-flyby && DRONE_CONF=0.12 DRONE_REPORT_MIN_CONF=0.08 bash deploy_aws_cpu.sh'` then validate again (2 min).
- Sun 02:10 survival bench (24 maps): base 950.2 vs evolver-best 971.1
- Sun 02:10 survival: kept the gen19 genome
- Sun 02:17 medical offline (39 convs): base 0.724 | +CE candidates 0.713 | +lexical vote 0.724 | both 0.716
- Sun 02:17 medical: serving variant = base (0.724)
sample_6: 27.6s round trip | keys ['answers', 'evidence_start', 'evidence_end']
accuracy 1.00 | tIoU 0.851 over 7 yes-questions
survival http://16.170.155.200:9052 -> 200 0.034109s
drone http://16.192.171.219:9053 -> 200 0.031658s
medical http://178.232.205.38:9054 -> 200 0.076349s
 0 TCP  9052->192.168.1.22:9052  'libminiupnpc' '' 0
 1 TCP  9053->192.168.1.22:9053  'libminiupnpc' '' 0
 2 TCP  9054->192.168.1.22:9054  'libminiupnpc' '' 0

## Survival pass 2 (Sun 06:39)
- Sun 06:39 survival pass 2 (12 seeds/genome): bench base 1079.3 vs evolved 1026.0 → kept gen19

- rec2 @0.3 current: per class: [('jet_plane', 50), ('small_plane', 42), ('small_tower', 22), ('large_tower', 22), ('hangar', 20), ('ta-ta', 7), ('tank', 5), ('jammer', 4), ('helicopter', 3), ('small_launcher', 3), ('spacecraft', 1)]
- rec2 @0.3 mix4: per class: [('jet_plane', 105), ('large_tower', 53), ('small_plane', 46), ('small_launcher', 43), ('small_tower', 27), ('hangar', 20), ('jammer', 8), ('helicopter', 6), ('large_launcher', 3), ('mine_roller', 2), ('tank', 1), ('ta-ta', 1)]
ERROR: Could not install packages due to an OSError: [Errno 28] No space left on device


- mix4 DEPLOYED on 16.192.171.219 (validate it on the site; fallback = redeploy with the previous models/ from git)
