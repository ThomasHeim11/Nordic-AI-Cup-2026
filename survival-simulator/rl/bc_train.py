"""Behaviour cloning: fit the actor to the heuristic's actions.

    python -m rl.bc_train --data rl/data/bc.npz --epochs 8 --out rl/weights/policy.pt
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import numpy as np
import torch
import torch.nn.functional as F

from rl.model import ActorCritic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="rl/data/bc.npz")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="rl/weights/policy.pt")
    a = ap.parse_args()
    d = np.load(a.data)
    X, Y = torch.from_numpy(d["X"]), torch.from_numpy(d["Y"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = ActorCritic().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    n = X.shape[0]
    print(f"{n} samples, device {dev}")
    for ep in range(a.epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, a.batch):
            idx = perm[i:i + a.batch]
            x, y = X[idx].to(dev), Y[idx].to(dev)
            mu, _, spawn_logit, _ = net(x)
            ang_pred, ang_true = mu[:, 1] * np.pi, y[:, 1] * np.pi     # direction is an angle: compare on the circle
            loss = (F.mse_loss(mu[:, 0], y[:, 0]) + (1 - torch.cos(ang_pred - ang_true)).mean()
                    + F.mse_loss(mu[:, 2], y[:, 2])
                    + F.binary_cross_entropy_with_logits(spawn_logit, y[:, 3], pos_weight=torch.tensor(5.0, device=dev)))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        print(f"epoch {ep}: loss {tot / n:.4f}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save(net.state_dict(), a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
