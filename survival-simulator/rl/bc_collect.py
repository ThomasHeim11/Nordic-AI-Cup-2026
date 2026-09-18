"""Collect (features, action) pairs from the heuristic hivemind for behaviour cloning.

    python -m rl.bc_collect --seeds 1000-1059 --workers 60 --out rl/data/bc.npz
"""
import argparse
import os
import random
import sys
from multiprocessing import Pool

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import numpy as np


def _episode(seed):
    import pygame
    pygame.init()
    from src.elements.environment import Environment
    Environment._render_biome_surface = lambda self: None
    from src.core import SimulationCore
    from src.utils.controllers import hivemind_policy as hp
    from rl.features import encode_all, action_to_vector
    hp.reset()
    sim = SimulationCore(seed=seed)
    rng = random.Random(seed)
    X, Y, actions = [], [], []
    while True:
        state = sim.step(actions)
        if state["num_agents"] == 0 or sim.env.time > 3000:
            break
        obs = state["observations"]
        acts = hp.decide_all(obs, rng)
        X.append(encode_all(obs, sim.env.time))
        Y.append(np.stack([action_to_vector(a, s) for a, s in zip(acts, obs)]))
        actions = [(a.agent_id, a) for a in acts]
    return np.concatenate(X), np.concatenate(Y), state["score"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1000-1011")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="rl/data/bc.npz")
    a = ap.parse_args()
    lo, hi = a.seeds.split("-")
    seeds = list(range(int(lo), int(hi) + 1))
    with Pool(a.workers) as pool:
        res = pool.map(_episode, seeds, chunksize=1)
    X = np.concatenate([r[0] for r in res])
    Y = np.concatenate([r[1] for r in res])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, X=X, Y=Y)
    print(f"saved {a.out}: {X.shape[0]} samples from {len(seeds)} episodes, teacher mean score {np.mean([r[2] for r in res]):.0f}")


if __name__ == "__main__":
    main()
