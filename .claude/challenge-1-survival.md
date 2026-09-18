# Challenge 1 — Survival Simulator (RL / simulation)

Folder: `survival-simulator/` · local port **9052** · endpoint `/predict`(see agent_server.py)

## The game
You are the hivemind of a herbivore species. Every tick you get the status+observations of
**every living agent**, and must return one `ActionRequest` per agent.
Sim runs until all agents die OR 3000 simulated seconds = **30 000 ticks** (=> 0.1 s/tick).
5 agents spawn at start. Predators spawn in over time with increasing odds and kill on touch.

## Observations (per agent)
`agent_id, observations, energy, biome, age, speed, sprint_speed, hearing_radius,
vision_angle, vision_range, max_energy`
Traits (speed, sprint_speed, hearing_radius, vision_angle, vision_range, max_energy) are
static per agent but **mutate slightly when spawning children** → evolution is a lever.

Observation types:
| type | data |
|---|---|
| Fruit | type, distance, angle (rad) |
| Agent | type, distance, angle, relative looking direction |
| Predator | type, distance, angle, relative looking direction |
| Tree | type, distance, angle |
| Edge | type, coordinates (start, end) |

Sensing = vision cone (vision_angle, vision_range, blocked by walls) ∪ hearing/smell radius
(omnidirectional, but **walls/edges are only ever seen, never heard**).

## Actions (per agent, per tick)
`agent_id:int, move_distance:float, move_direction:float (absolute rad),
 turn_angle:float (rad), spawn_agent:bool`

## Energy economy — the core of the whole challenge
| action | cost |
|---|---|
| walk (move_distance <= speed) | `move_distance * 0.05` |
| sprint (speed < d <= sprint_speed) | `speed*0.05 + (d - speed)*0.5` |
| turn | `abs(turn_angle) / 2*pi`  (as written in README) |
| spawn | **100** |
| living (passive) | `1/10 * biome_energy_modifier` per tick |
| ageing | past a random age in [60,120] s, living cost += `0.01 * age` |

Sprinting is **10x** the per-unit cost of walking → only sprint to escape predators.
Ageing means old agents get expensive: breeding a replacement (100) vs. letting one starve
is a real trade-off.

## Score
Mainly **survival time of the species**. Then: +small per fruit eaten,
**− based on the remaining energy of any agent eaten by a predator**.
=> A starving agent that dies of hunger costs less than a fat agent that gets eaten.
=> Never let a high-energy agent be near a predator; low-energy agents are cheap bait.

## Files
- `local_playground.py` — headless/verbose local sim. Use `verbose=False` for training speed.
- `simulation_server.py` — runs a sim against your served endpoint.
- `agent_server.py` — the FastAPI server to submit (HOST/PORT editable).
- `src/utils/controllers/dummy_agent_policy.py` — baseline policy to replace.
- `src/utils/DTOs.py` — request/response models.

## Gotchas
- Sim is deterministic **per OS** — test seeds on Linux if you want to match the server.
- Server waits max **10 s** per response; total accumulated wait 600 s ends the run.
- Known repo bug: `DTOs.py` may need `sim_time: float = 0.0` and `n_agents: int = 0`
  or the test endpoint 422s. Check it is patched.
- **Evaluation = 3 runs back to back, averaged.** The agent server must survive all three
  (no state leaking between runs; reset cleanly on a new sim).
