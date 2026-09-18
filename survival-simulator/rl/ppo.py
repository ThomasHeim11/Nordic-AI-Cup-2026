"""PPO for the survival hivemind: many simulators, one shared actor-critic.

    python -m rl.ppo --envs 60 --rollout 256 --iters 400 --init rl/weights/policy.pt --out rl/weights/ppo

Every animal in every simulator is one row.  Per tick each animal gets
    r = 0.1                        (the score itself: +dt while the species lives)
      + 0.002 * energy gained      (eating; breeding costs are visible as losses)
      - 1.0 on death               (eaten or starved -> its trajectory ends)
Advantages are computed per animal over its own lifetime (GAE, gamma 0.999),
bootstrapped with the critic where a rollout cuts a life short.  The update is
the clipped PPO objective on the summed log-probability of the Gaussian move
head and the Bernoulli breed head, plus value loss and an entropy bonus.

Checkpoints go to <out>/iter_XXXX.pt; evaluate any of them with
    SURVIVAL_NN=<ckpt> python bench_many.py --seeds 201-224 --workers 8 ...  (policy rl.policy_nn)
"""
import argparse
import os
import random
import sys
import time
from multiprocessing import Pipe, Process

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import numpy as np
import torch

from rl.features import FEATURE_DIM, encode_all, vector_to_action
from rl.model import ActorCritic

R_ALIVE, R_ENERGY, R_DEATH = 0.1, 0.002, 1.0


# --------------------------------------------------------------------------- #
# Simulator worker
# --------------------------------------------------------------------------- #

def _worker(conn, seed0: int):
    import pygame
    pygame.init()
    from src.elements.environment import Environment
    Environment._render_biome_surface = lambda self: None
    from src.core import SimulationCore

    rng = random.Random(seed0)
    sim, state, prev_energy, episode_score = None, None, {}, 0.0

    def new_episode():
        nonlocal sim, state, prev_energy
        sim = SimulationCore(seed=rng.randint(1, 10 ** 9))
        state = sim.step([])
        prev_energy = {s["agent_id"]: s["energy"] for s in state["observations"]}

    new_episode()
    conn.send(("obs", state["observations"], sim.env.time))
    while True:
        msg = conn.recv()
        if msg is None:
            break
        actions = [(a.agent_id, a) for a in msg]              # ActionRequest list
        state = sim.step(actions)
        obs = state["observations"]
        alive = {s["agent_id"]: s["energy"] for s in obs}
        rewards, dones = {}, {}
        for aid, e0 in prev_energy.items():
            if aid in alive:
                rewards[aid] = R_ALIVE + R_ENERGY * max(0.0, alive[aid] - e0)
                dones[aid] = False
            else:
                rewards[aid] = -R_DEATH
                dones[aid] = True
        episode_done = state["num_agents"] == 0 or sim.env.time > 3000
        score = state["score"]
        prev_energy = alive
        if episode_done:
            new_episode()                      # sets state/prev_energy for the fresh simulator
            obs = state["observations"]
        conn.send(("step", obs, sim.env.time, rewards, dones, episode_done, score))
    conn.close()


class VecSim:
    def __init__(self, n: int, seed: int = 0):
        self.conns, self.procs = [], []
        for i in range(n):
            a, b = Pipe()
            p = Process(target=_worker, args=(b, seed * 1000 + i), daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)
        self.obs, self.t = [], []
        for c in self.conns:
            tag, obs, t = c.recv()
            self.obs.append(obs)
            self.t.append(t)

    def step(self, actions_per_env):
        for c, acts in zip(self.conns, actions_per_env):
            c.send(acts)
        out = [c.recv() for c in self.conns]
        self.obs = [o[1] for o in out]
        self.t = [o[2] for o in out]
        return out

    def close(self):
        for c in self.conns:
            try:
                c.send(None)
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)


# --------------------------------------------------------------------------- #
# Rollout storage: one trajectory per (env, agent)
# --------------------------------------------------------------------------- #

class Traj:
    __slots__ = ("x", "a", "logp", "v", "r", "done")

    def __init__(self):
        self.x, self.a, self.logp, self.v, self.r, self.done = [], [], [], [], [], False


def log_prob(normal, bern, a):
    return normal.log_prob(a[:, :3]).sum(-1) + bern.log_prob(a[:, 3])


def gae(rewards, values, last_value, gamma, lam):
    adv = np.zeros(len(rewards), np.float32)
    g = 0.0
    for i in reversed(range(len(rewards))):
        nv = last_value if i == len(rewards) - 1 else values[i + 1]
        delta = rewards[i] + gamma * nv - values[i]
        g = delta + gamma * lam * g
        adv[i] = g
    return adv, adv + np.asarray(values, np.float32)


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--rollout", type=int, default=256, help="ticks per rollout")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--init", default="rl/weights/policy.pt", help="behaviour-cloned starting point")
    ap.add_argument("--out", default="rl/weights/ppo")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--lam", type=float, default=0.97)
    ap.add_argument("--clip", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--minibatch", type=int, default=8192)
    ap.add_argument("--entropy", type=float, default=0.001)
    ap.add_argument("--ckpt-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = ActorCritic().to(dev)
    if a.init and os.path.exists(a.init):
        net.load_state_dict(torch.load(a.init, map_location=dev))
        print("initialised from", a.init)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    os.makedirs(a.out, exist_ok=True)
    vec = VecSim(a.envs, a.seed)
    scores = []
    t_start = time.time()

    for it in range(a.iters):
        trajs = {}                                   # (env, agent) -> Traj
        cur = [dict() for _ in range(a.envs)]        # per env: agent -> key of live traj
        for step in range(a.rollout):
            # ---- act for every animal in every env in one batch
            feats, index = [], []
            for e in range(a.envs):
                obs = vec.obs[e]
                if obs:
                    feats.append(encode_all(obs, vec.t[e]))
                    index += [(e, s["agent_id"]) for s in obs]
            if not index:
                vec.step([[] for _ in range(a.envs)])
                continue
            x = torch.from_numpy(np.concatenate(feats)).to(dev)
            with torch.no_grad():
                normal, bern, value = net.dist(x)
                cont = normal.sample()
                spawn = bern.sample()
                act = torch.cat([cont, spawn.unsqueeze(-1)], dim=-1)
                lp = log_prob(normal, bern, act)
            act_np, lp_np, v_np, x_np = act.cpu().numpy(), lp.cpu().numpy(), value.cpu().numpy(), x.cpu().numpy()
            actions_per_env = [[] for _ in range(a.envs)]
            row_of = {}
            for i, (e, aid) in enumerate(index):
                st = next(s for s in vec.obs[e] if s["agent_id"] == aid)
                actions_per_env[e].append(vector_to_action(act_np[i], st))
                row_of[(e, aid)] = i
            # ---- step all simulators
            out = vec.step(actions_per_env)
            # ---- record transitions
            for (e, aid), i in row_of.items():
                key = cur[e].get(aid)
                if key is None or trajs[key].done:
                    key = (e, aid, it, step)
                    cur[e][aid] = key
                    trajs[key] = Traj()
                tr = trajs[key]
                tr.x.append(x_np[i]); tr.a.append(act_np[i]); tr.logp.append(lp_np[i]); tr.v.append(v_np[i])
                rewards, dones, ep_done, score = out[e][3], out[e][4], out[e][5], out[e][6]
                tr.r.append(rewards.get(aid, 0.0))
                if dones.get(aid, False) or ep_done:
                    tr.done = True
                if ep_done:
                    scores.append(score)
            for e in range(a.envs):
                if out[e][5]:
                    cur[e] = {}
        # ---- bootstrap live trajectories with the critic on the latest observations
        last_v = {}
        for e in range(a.envs):
            obs = vec.obs[e]
            if obs:
                with torch.no_grad():
                    v = net(torch.from_numpy(encode_all(obs, vec.t[e])).to(dev))[3].cpu().numpy()
                for s, vv in zip(obs, v):
                    last_v[(e, s["agent_id"])] = float(vv)
        X, A, LP, ADV, RET = [], [], [], [], []
        for (e, aid, _, _), tr in trajs.items():
            if not tr.r:
                continue
            boot = 0.0 if tr.done else last_v.get((e, aid), 0.0)
            adv, ret = gae(tr.r, tr.v, boot, a.gamma, a.lam)
            X.append(np.stack(tr.x)); A.append(np.stack(tr.a)); LP.append(np.asarray(tr.logp)); ADV.append(adv); RET.append(ret)
        X = torch.from_numpy(np.concatenate(X)).to(dev)
        A = torch.from_numpy(np.concatenate(A)).to(dev)
        LP = torch.from_numpy(np.concatenate(LP)).to(dev)
        ADV = torch.from_numpy(np.concatenate(ADV)).to(dev)
        RET = torch.from_numpy(np.concatenate(RET)).to(dev)
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-6)
        n = X.shape[0]
        # ---- PPO update
        stats = []
        for _ in range(a.epochs):
            perm = torch.randperm(n, device=dev)
            for i in range(0, n, a.minibatch):
                idx = perm[i:i + a.minibatch]
                normal, bern, value = net.dist(X[idx])
                lp = log_prob(normal, bern, A[idx])
                ratio = torch.exp(lp - LP[idx])
                pg = -torch.min(ratio * ADV[idx], torch.clamp(ratio, 1 - a.clip, 1 + a.clip) * ADV[idx]).mean()
                vl = 0.5 * (value - RET[idx]).pow(2).mean()
                ent = (normal.entropy().sum(-1) + bern.entropy()).mean()
                loss = pg + 0.5 * vl - a.entropy * ent
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
                stats.append((pg.item(), vl.item(), ent.item()))
        pg, vl, ent = np.mean(stats, axis=0)
        recent = scores[-20:]
        print(f"iter {it:4d} | rows {n:7d} | pg {pg:+.4f} vl {vl:.3f} ent {ent:.3f} | "
              f"episodes {len(scores):4d} recent mean score {np.mean(recent) if recent else 0:7.1f} | {time.time() - t_start:6.0f}s", flush=True)
        if (it + 1) % a.ckpt_every == 0:
            torch.save(net.state_dict(), os.path.join(a.out, f"iter_{it + 1:04d}.pt"))
    torch.save(net.state_dict(), os.path.join(a.out, "final.pt"))
    vec.close()


if __name__ == "__main__":
    main()
