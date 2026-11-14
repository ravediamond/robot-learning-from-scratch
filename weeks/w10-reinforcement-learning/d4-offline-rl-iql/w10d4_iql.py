"""W10 D4: behavior cloning vs IQL on a mixed-quality dataset."""
import copy

import torch
from torch import nn


def mlp(inp: int, out: int, hidden: int = 128) -> nn.Module:
    return nn.Sequential(nn.Linear(inp, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out))


def expectile_loss(diff: torch.Tensor, tau: float) -> torch.Tensor:
    """mean(|tau - 1[diff < 0]| * diff^2)."""
    # TODO
    raise NotImplementedError


def awr_weights(adv: torch.Tensor, beta: float, w_max: float = 100.0) -> torch.Tensor:
    """min(exp(beta * adv), w_max), with no gradient."""
    # TODO
    raise NotImplementedError


def train_bc(data: dict, steps: int = 1500, batch: int = 256, lr: float = 1e-3, seed: int = 0):
    """MSE regression of act on obs with an mlp(4, 2). Return act_fn(obs) -> action (no grad)."""
    # TODO
    raise NotImplementedError


def train_iql(data: dict, steps: int = 2500, batch: int = 256, lr: float = 1e-3, gamma: float = 0.95,
              tau: float = 0.8, beta: float = 5.0, polyak: float = 0.01, seed: int = 0):
    """IQL with one Q net (plus a slow target copy), one V net, one deterministic policy.
    Per step, on a random batch (s, a, r, s', done):
      V:  expectile_loss(Q_target(s, a) - V(s), tau)                       (Q_target has no grad)
      Q:  mean((Q(s, a) - (r + gamma * (1 - done) * V(s')))^2)             (the target has no grad)
      pi: mean(awr_weights(Q_target(s, a) - V(s), beta) * ||pi(s) - a||^2) (sum over action dims)
      Q_target <- polyak update.
    Return act_fn(obs) -> action (no grad)."""
    # TODO
    raise NotImplementedError
