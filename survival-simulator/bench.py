"""Headless benchmark harness for the survival simulator.

    python bench.py --policy src.utils.controllers.hivemind_policy --seeds 1,2,3

``--fast`` skips Environment._render_biome_surface, a 1.92M-call per-pixel loop
that costs ~11 s per environment and only exists for the pygame view.  It draws
from the same RNG stream, so a --fast world is NOT the same world as the same
seed without it -- it is still a fair sample of the same distribution, which is
all a benchmark needs.  Leave it off when you care about a specific seed.
"""
import argparse, importlib, os, random, statistics, time
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame
pygame.init()
from src.core import SimulationCore
from src.elements.environment import Environment


def patch_fast_biome_render():
    Environment._render_biome_surface = lambda self: None


DEATHS = {"starved": 0, "eaten": 0, "eaten_energy": 0.0, "ages": []}


def patch_death_counter():
    orig = Environment.kill_agent

    def kill_agent(self, agent):
        if agent in self.agents:
            if agent.energy <= 0:
                DEATHS["starved"] += 1
            else:
                DEATHS["eaten"] += 1
                DEATHS["eaten_energy"] += agent.energy
            DEATHS["ages"].append(agent.age)
        return orig(self, agent)
    Environment.kill_agent = kill_agent


def run_one(policy, seed, max_time=3000.0, log_every=0):
    for k in ("starved", "eaten"):
        DEATHS[k] = 0
    DEATHS["eaten_energy"] = 0.0
    DEATHS["ages"] = []
    sim = SimulationCore(seed=seed)
    rng = random.Random(seed)
    if hasattr(policy, "reset"):
        policy.reset()
    batch = getattr(policy, "decide_all", None)
    single = getattr(policy, "action_decision", None)

    actions, state = [], None
    t0 = time.perf_counter()
    next_log = 0.0
    peak_pop = 0
    while True:
        state = sim.step(actions)
        peak_pop = max(peak_pop, state["num_agents"])
        if state["num_agents"] == 0 or sim.env.time > max_time:
            break
        obs = state["observations"]
        if batch is not None:
            actions = [(a.agent_id, a) for a in batch(obs, rng)]
        else:
            actions = [(s["agent_id"], single(s, rng)) for s in obs]
        if log_every and sim.env.time >= next_log:
            next_log += log_every
            en = [a.energy for a in sim.env.agents]
            print(f"  t={sim.env.time:7.1f} score={state['score']:8.2f} "
                  f"agents={state['num_agents']:3d} fruits={len(sim.env.fruits):4d} "
                  f"trees={len(sim.env.trees):3d} preds={len(sim.env.predators):3d} "
                  f"energy mean={statistics.mean(en):6.1f} max={max(en):6.1f} "
                  f"starved={DEATHS['starved']} eaten={DEATHS['eaten']}", flush=True)
    return {"seed": seed, "score": state["score"], "time": sim.env.time,
            "agents": state["num_agents"], "peak_pop": peak_pop,
            "starved": DEATHS["starved"], "eaten": DEATHS["eaten"],
            "eaten_energy": DEATHS["eaten_energy"],
            "mean_age": statistics.mean(DEATHS["ages"]) if DEATHS["ages"] else 0.0,
            "wall": time.perf_counter() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="src.utils.controllers.dummy_agent_policy")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--max-time", type=float, default=3000.0)
    ap.add_argument("--log-every", type=float, default=0)
    ap.add_argument("--fast", action="store_true", help="skip the biome pixel render")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a PARAMS entry in the policy")
    a = ap.parse_args()

    if a.fast:
        patch_fast_biome_render()
    patch_death_counter()
    policy = importlib.import_module(a.policy)
    for kv in a.set:
        k, v = kv.split("=", 1)
        policy.PARAMS[k] = float(v)

    results = []
    for seed in [int(s) for s in a.seeds.split(",")]:
        r = run_one(policy, seed, a.max_time, a.log_every)
        results.append(r)
        print(f"seed {r['seed']:>10} | score {r['score']:9.2f} | survived {r['time']:7.1f}s "
              f"| left {r['agents']:3d} | peak {r['peak_pop']:3d} | starved {r['starved']:3d} "
              f"| eaten {r['eaten']:3d} (-{r['eaten_energy']/100:5.1f}) | mean age {r['mean_age']:5.1f} "
              f"| wall {r['wall']:6.1f}s", flush=True)

    scores = [r["score"] for r in results]
    line = f"  mean score {statistics.mean(scores):.2f}"
    if len(scores) > 1:
        line += f"  stdev {statistics.stdev(scores):.2f}"
    print(f"\n{a.policy}\n{line}  min {min(scores):.2f}  max {max(scores):.2f}")


if __name__ == "__main__":
    main()
