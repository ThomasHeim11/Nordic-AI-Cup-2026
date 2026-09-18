"""Fixed-size egocentric feature vector for one animal.

Everything the hand-written policy looks at, flattened: own state and traits,
the K nearest fruits / trees / mates / predators (distance, direction, facing),
the closest wall point, and a few hivemind-level numbers (time, population,
food per capita).  Angles are relative to the animal's heading, exactly as the
simulator reports them, so the network never needs a global frame.
"""
import math
from typing import Dict, List, Sequence

import numpy as np

BIOMES = ["forest", "grassland", "swamp", "desert", "river"]
K_FRUIT, K_TREE, K_MATE, K_PRED = 4, 3, 3, 3
DIST_SCALE = 400.0

FEATURE_DIM = 13 + len(BIOMES) + 3 + 3 * K_FRUIT + 3 * K_TREE + 5 * K_MATE + 5 * K_PRED + 4 + 3


def _closest_point_on_edge(coords):
    (x1, y1), (x2, y2) = coords
    dx, dy = x2 - x1, y2 - y1
    d = dx * dx + dy * dy
    if d <= 1e-9:
        return x1, y1
    t = max(0.0, min(1.0, -(x1 * dx + y1 * dy) / d))
    return x1 + t * dx, y1 + t * dy


def encode(state: Dict, t_frac: float, pop: int, food_index: float) -> np.ndarray:
    f: List[float] = []
    me, sp, ss = state["max_energy"], state["speed"], state["sprint_speed"]
    f += [state["energy"] / max(me, 1.0), state["energy"] / 500.0, min(state["age"], 200.0) / 120.0,
          sp / 20.0, ss / 40.0, state["hearing_radius"] / 100.0, state["vision_range"] / 400.0,
          state["vision_angle"] / (math.pi / 2), me / 1000.0, t_frac, min(pop, 60) / 40.0,
          min(food_index, 10.0) / 5.0, 1.0]
    f += [1.0 if state.get("biome") == b else 0.0 for b in BIOMES]
    obs = state["observations"]
    fruits = sorted((o for o in obs if o.get("type") == "Fruit"), key=lambda o: o["distance"])
    trees = sorted((o for o in obs if o.get("type") == "Tree"), key=lambda o: o["distance"])
    mates = sorted((o for o in obs if o.get("type") == "Agent"), key=lambda o: o["distance"])
    preds = sorted((o for o in obs if o.get("type") == "Predator"), key=lambda o: o["distance"])
    edges = [o for o in obs if o.get("type") == "Edge"]
    f += [min(len(fruits), 20) / 10.0, min(len(trees), 20) / 10.0, min(len(preds), 10) / 5.0]

    def slot3(o):
        return [o["distance"] / DIST_SCALE, math.cos(o["angle"]), math.sin(o["angle"])] if o else [2.0, 0.0, 0.0]

    def slot5(o):
        return slot3(o) + ([math.cos(o.get("rel_dir", 0.0)), math.sin(o.get("rel_dir", 0.0))] if o else [0.0, 0.0])

    for i in range(K_FRUIT):
        f += slot3(fruits[i] if i < len(fruits) else None)
    for i in range(K_TREE):
        f += slot3(trees[i] if i < len(trees) else None)
    for i in range(K_MATE):
        f += slot5(mates[i] if i < len(mates) else None)
    for i in range(K_PRED):
        f += slot5(preds[i] if i < len(preds) else None)
    f += [min(len(mates), 20) / 10.0, 1.0 if fruits else 0.0, 1.0 if trees else 0.0, 1.0 if preds else 0.0]
    if edges:
        cx, cy = min((_closest_point_on_edge(e["coords"]) for e in edges), key=lambda q: math.hypot(*q))
        d = math.hypot(cx, cy)
        f += [min(d, 200.0) / 100.0, cx / max(d, 1e-6), cy / max(d, 1e-6)]
    else:
        f += [2.0, 0.0, 0.0]
    v = np.asarray(f, dtype=np.float32)
    assert v.shape[0] == FEATURE_DIM, (v.shape, FEATURE_DIM)
    return v


def encode_all(states: Sequence[Dict], t: float) -> np.ndarray:
    pop = len(states)
    if not states:
        return np.zeros((0, FEATURE_DIM), np.float32)
    n_fruit = sum(1 for s in states for o in s["observations"] if o.get("type") == "Fruit")
    n_tree = sum(1 for s in states for o in s["observations"] if o.get("type") == "Tree")
    food_index = (n_fruit + 0.5 * n_tree) / max(pop, 1)
    return np.stack([encode(s, t / 3000.0, pop, food_index) for s in states])


def action_to_vector(a, state: Dict) -> np.ndarray:
    """ActionRequest -> [dist_frac, move_dir/pi, turn/pi, spawn] (the network's action space)."""
    return np.array([min(a.move_distance / max(state["sprint_speed"], 1e-6), 1.0),
                     ((a.move_direction + math.pi) % (2 * math.pi) - math.pi) / math.pi,
                     max(-1.0, min(1.0, a.turn_angle / math.pi)),
                     1.0 if a.spawn_agent else 0.0], dtype=np.float32)


def vector_to_action(v: np.ndarray, state: Dict):
    from src.utils.DTOs import ActionRequest
    dist = float(np.clip(v[0], 0.0, 1.0)) * state["sprint_speed"]
    return ActionRequest(agent_id=state["agent_id"], move_distance=dist,
                         move_direction=float(np.clip(v[1], -1.0, 1.0)) * math.pi,
                         turn_angle=float(np.clip(v[2], -1.0, 1.0)) * math.pi,
                         spawn_agent=bool(v[3] > 0.5))
