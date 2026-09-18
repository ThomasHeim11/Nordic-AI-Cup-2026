"""Network-backed policy with the same interface as hivemind_policy (decide_all / reset)."""
import os

import torch

from rl.features import encode_all, vector_to_action
from rl.model import ActorCritic

WEIGHTS = os.environ.get("SURVIVAL_NN", os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "policy.pt"))
_NET = {"m": None}
_T = {"t": 0.0}


def reset():
    _T["t"] = 0.0


def load(path: str = WEIGHTS):
    if _NET["m"] is None:
        m = ActorCritic()
        m.load_state_dict(torch.load(path, map_location="cpu"))
        m.eval()
        torch.set_num_threads(1)
        _NET["m"] = m
    return _NET["m"]


def decide_all(agent_states, rng=None):
    states = [s for s in agent_states if s is not None]
    if not states:
        return []
    net = load()
    x = torch.from_numpy(encode_all(states, _T["t"]))
    v, _ = net.act(x, deterministic=True)
    _T["t"] += 0.1
    v = v.numpy()
    return [vector_to_action(v[i], s) for i, s in enumerate(states)]
