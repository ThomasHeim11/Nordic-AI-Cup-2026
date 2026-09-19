"""Per-tick latency of a /predict endpoint, the way the evaluator sees it.

Builds one genuine step (real observations after 60 simulated ticks) and posts
it N times, opening a fresh TCP connection every time (the evaluator does not
reuse connections).  Prints median / p90 / max wall time per request.

    python latency_probe.py http://127.0.0.1:9052 --n 30
"""
import argparse, json, os, statistics, sys, time
os.environ.setdefault("SDL_VIDEODRIVER", "dummy"); os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import requests


def build_step(ticks=60):
    import pygame; pygame.init()
    from src.elements.environment import Environment
    Environment._render_biome_surface = lambda self: None
    from src.core import SimulationCore
    from src.utils.controllers import hivemind_policy
    import random; rng = random.Random(1)
    sim = SimulationCore(seed=7); state = sim.step([])
    for _ in range(ticks):
        acts = hivemind_policy.decide_all(state["observations"], rng)
        state = sim.step([(a.agent_id, a) for a in acts])
    return {"game_status": "running", "score": state["score"], "sim_time": state["sim_time"],
            "n_agents": state["num_agents"], "agent_status": state["observations"]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("url"); ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    step = build_step(); body = json.dumps(step)
    url = a.url.rstrip("/") + "/predict"
    print(f"payload {len(body)/1024:.1f} kB, {step['n_agents']} agents -> {url}")
    ts = []
    for i in range(a.n):
        t0 = time.perf_counter()
        r = requests.post(url, data=body, headers={"content-type": "application/json", "connection": "close"}, timeout=30)
        ts.append(time.perf_counter() - t0)
        if r.status_code != 200:
            print("status", r.status_code, r.text[:200]); break
    ts.sort()
    print(f"n={len(ts)} median {statistics.median(ts)*1000:.0f} ms | p90 {ts[int(0.9*len(ts))-1]*1000:.0f} ms | max {ts[-1]*1000:.0f} ms | rule needs < 20 ms avg incl. simulator side")


if __name__ == "__main__":
    main()
