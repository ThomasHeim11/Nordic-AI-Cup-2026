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
| path | ms/tick (from the 600 s rule) | validation |
|---|---|---|
| Mac + Cloudflare tunnel | 97 | 619 s (cut) |
| cluster n009 + Cloudflare tunnel | 60 | 992 s (cut) |
| Mac direct, router port-forward (UPnP), Wi-Fi | 80 | 750 s (cut) |

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
A small elite population is good for *both* survival and latency, so B1 and hosting pull the same way.

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
- Cluster evolver / CMA-ES on the final behaviour, 8–16 maps per candidate, optimise the *mean*.

## Also in the plan (drone, from Discord): estimate camera motion online from consecutive frames
instead of the Helsinki-fitted homography — the evaluation flight direction may differ.

## Timeline
| When | Owner | Step |
|---|---|---|
| Fri night | done | cluster + home hosting measured (60 / 80 ms per tick) |
| Fri 23:50 | done | B1 endgame rewrite: 1002 mean on 24 maps (was 950), variance down |
| Fri 23:50 | done | B2 code complete and smoke-tested (features, actor-critic, BC, PPO, cluster runner) |
| Sat midday | Claude | RL groundwork: env wrapper, features, BC data + training on A40; go/no-go |
| Sat afternoon → Sun morning | cluster | PPO, hourly checkpoints scored on 24 maps |
| Sat afternoon | both | hosting: small-population run, delayed-ACK off, Ethernet if available; re-validate |
| Sun 09–12 | both | pick best policy, validate on the real scoreboard, freeze |
| Sun ~14 | you | final evaluation (one attempt) |

## Definition of done
- Validation ≥ 1800 s from the final host, repeated twice.
- Mean over 24 local maps ≥ 1900 s.
- Training code + model saved in the repo (top-5 must hand it in by Sun 20:00).
