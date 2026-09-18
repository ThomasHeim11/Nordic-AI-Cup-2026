"""Shared actor-critic network: one net, every animal is a row of the batch.

Actor heads: Gaussian over (distance fraction, move direction / pi, turn / pi)
and a Bernoulli for "breed".  Critic head: state value.  Small enough to run
in well under a millisecond per tick on a laptop CPU, which matters because
the evaluator's clock counts our compute.
"""
import torch
import torch.nn as nn

from rl.features import FEATURE_DIM


class ActorCritic(nn.Module):
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(FEATURE_DIM, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.mu = nn.Linear(hidden, 3)
        self.log_std = nn.Parameter(torch.full((3,), -1.0))
        self.spawn_logit = nn.Linear(hidden, 1)
        self.value = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.trunk(x)
        mu = self.mu(h)
        return mu, self.log_std.expand_as(mu), self.spawn_logit(h).squeeze(-1), self.value(h).squeeze(-1)

    def dist(self, x):
        mu, log_std, spawn_logit, value = self.forward(x)
        return torch.distributions.Normal(mu, log_std.exp()), torch.distributions.Bernoulli(logits=spawn_logit), value

    @torch.no_grad()
    def act(self, x, deterministic: bool = True):
        normal, bern, value = self.dist(x)
        cont = normal.mean if deterministic else normal.sample()
        spawn = (bern.probs > 0.5).float() if deterministic else bern.sample()
        return torch.cat([cont, spawn.unsqueeze(-1)], dim=-1), value
