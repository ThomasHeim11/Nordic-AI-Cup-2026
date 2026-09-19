# TODO — Nordic AI Cup 2026 — target: 1st in all three
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
