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
- [ ] `drone_exp.log` (job tmp dir): conf 0.12 vs 0.05, flip TTA, ta-ta 0.5 on Helsinki → if one wins, set the env var
      (`DRONE_CONF`, `DRONE_TTA=1`, `DRONE_CLASS_CONF='{"ta-ta":0.5}'`) and restart :9053.
- [ ] `local_evaluator.py --realtime --url http://178.232.205.38:9053/...` (25/25 frames, < 250 ms) → **validate** → read the new recording.
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
