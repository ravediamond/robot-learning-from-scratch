"""W10 D4: behavior cloning vs IQL on a mixed-quality dataset."""
import copy

import torch
from torch import nn


def mlp(inp: int, out: int, hidden: int = 128) -> nn.Module:
    return nn.Sequential(nn.Linear(inp, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out))


def expectile_loss(diff: torch.Tensor, tau: float) -> torch.Tensor:
    """mean(|tau - 1[diff < 0]| * diff^2)."""
    w = torch.where(diff < 0, 1 - tau, tau)
    return (w * diff.pow(2)).mean()


def awr_weights(adv: torch.Tensor, beta: float, w_max: float = 100.0) -> torch.Tensor:
    """min(exp(beta * adv), w_max), with no gradient."""
    return (beta * adv.detach()).exp().clamp(max=w_max)


def train_bc(data: dict, steps: int = 1500, batch: int = 256, lr: float = 1e-3, seed: int = 0):
    """MSE regression of act on obs with an mlp(4, 2). Return act_fn(obs) -> action (no grad)."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    pi = mlp(4, 2)
    opt = torch.optim.Adam(pi.parameters(), lr=lr)
    n = len(data["obs"])
    for _ in range(steps):
        idx = torch.randint(0, n, (batch,), generator=g)
        loss = (pi(data["obs"][idx]) - data["act"][idx]).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return lambda obs: pi(obs).detach()


def train_iql(data: dict, steps: int = 2500, batch: int = 256, lr: float = 1e-3, gamma: float = 0.95,
              tau: float = 0.8, beta: float = 5.0, polyak: float = 0.01, seed: int = 0):
    """IQL with one Q net (plus a slow target copy), one V net, one deterministic policy.
    Per step, on a random batch (s, a, r, s', done):
      V:  expectile_loss(Q_target(s, a) - V(s), tau)                       (Q_target has no grad)
      Q:  mean((Q(s, a) - (r + gamma * (1 - done) * V(s')))^2)             (the target has no grad)
      pi: mean(awr_weights(Q_target(s, a) - V(s), beta) * ||pi(s) - a||^2) (sum over action dims)
      Q_target <- polyak update.
    Return act_fn(obs) -> action (no grad)."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    q, v, pi = mlp(6, 1), mlp(4, 1), mlp(4, 2)
    q_t = copy.deepcopy(q)
    opt = torch.optim.Adam(list(q.parameters()) + list(v.parameters()) + list(pi.parameters()), lr=lr)
    n = len(data["obs"])
    for _ in range(steps):
        idx = torch.randint(0, n, (batch,), generator=g)
        s, a, r, s2, d = (data[k][idx] for k in ("obs", "act", "rew", "next_obs", "done"))
        sa = torch.cat([s, a], dim=-1)
        with torch.no_grad():
            q_tgt = q_t(sa).squeeze(-1)                                  # (B,)
            y = r + gamma * (~d).float() * v(s2).squeeze(-1)              # (B,)
        v_s = v(s).squeeze(-1)
        loss_v = expectile_loss(q_tgt - v_s, tau)
        loss_q = (q(sa).squeeze(-1) - y).pow(2).mean()
        w = awr_weights(q_tgt - v_s, beta)
        loss_pi = (w * (pi(s) - a).pow(2).sum(dim=-1)).mean()
        opt.zero_grad()
        (loss_v + loss_q + loss_pi).backward()
        opt.step()
        with torch.no_grad():
            for pt, ps in zip(q_t.parameters(), q.parameters()):
                pt.mul_(1 - polyak).add_(polyak * ps)
    return lambda obs: pi(obs).detach()
