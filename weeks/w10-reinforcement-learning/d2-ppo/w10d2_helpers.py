"""Provided: yesterday's pieces, so today stands alone."""
import torch
from torch import nn


def gae(rewards, values, dones, last_value, gamma, lam):
    adv = torch.zeros_like(rewards)
    nxt_adv, nxt_val = torch.zeros_like(last_value), last_value
    for t in range(rewards.shape[0] - 1, -1, -1):
        nd = (~dones[t]).float()
        delta = rewards[t] + gamma * nd * nxt_val - values[t]
        nxt_adv = delta + gamma * lam * nd * nxt_adv
        adv[t] = nxt_adv
        nxt_val = values[t]
    return adv


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, act_dim))
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def dist(self, obs):
        return torch.distributions.Normal(self.net(obs), self.log_std.exp())

    def log_prob(self, obs, action):
        return self.dist(obs).log_prob(action).sum(dim=-1)


def value_net(obs_dim, hidden=64):
    return nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                         nn.Linear(hidden, 1))


@torch.no_grad()
def collect(env, policy, value):
    """Full episodes. Returns a dict of time-major tensors, all WITHOUT grad:
    obs (T, N, D), act (T, N, A), logp (T, N), rew (T, N), done (T, N), val (T, N), last_val (N,)."""
    obs = env.reset()
    out = {k: [] for k in ("obs", "act", "logp", "rew", "done", "val")}
    for _ in range(env.horizon):
        a = policy.dist(obs).sample()
        out["obs"].append(obs)
        out["act"].append(a)
        out["logp"].append(policy.log_prob(obs, a))
        out["val"].append(value(obs).squeeze(-1))
        obs, r, d = env.step(a)
        out["rew"].append(r)
        out["done"].append(d)
    batch = {k: torch.stack(v) for k, v in out.items()}
    batch["last_val"] = value(obs).squeeze(-1)
    return batch
