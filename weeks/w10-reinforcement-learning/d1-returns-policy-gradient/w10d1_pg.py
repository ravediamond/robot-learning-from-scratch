"""W10 D1: returns, GAE, REINFORCE. Time-major tensors: (T, N)."""
import torch
from torch import nn

import rl_envs


def discounted_returns(rewards: torch.Tensor, dones: torch.Tensor, gamma: float) -> torch.Tensor:
    """rewards (T, N), dones (T, N) bool (episode ended AT step t) -> G (T, N).
    G_t = r_t + gamma * (1 - done_t) * G_{t+1}, with G_T = 0 after the last stored step."""
    T = rewards.shape[0]
    G = torch.zeros_like(rewards)
    nxt = torch.zeros_like(rewards[0])  # (N,)
    for t in range(T - 1, -1, -1):
        nxt = rewards[t] + gamma * (~dones[t]).float() * nxt
        G[t] = nxt
    return G


def gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, last_value: torch.Tensor,
        gamma: float, lam: float) -> torch.Tensor:
    """rewards, values, dones (T, N); last_value (N,) = V of the state after the last step. -> advantages (T, N)."""
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    nxt_adv = torch.zeros_like(last_value)
    nxt_val = last_value
    for t in range(T - 1, -1, -1):
        nd = (~dones[t]).float()
        delta = rewards[t] + gamma * nd * nxt_val - values[t]  # (N,)
        nxt_adv = delta + gamma * lam * nd * nxt_adv
        adv[t] = nxt_adv
        nxt_val = values[t]
    return adv


class GaussianPolicy(nn.Module):
    """Diagonal Gaussian: mean from an MLP, one learned log-std per action dimension."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, act_dim))
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        return torch.distributions.Normal(self.net(obs), self.log_std.exp())

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """log pi(a | s), SUMMED over action dimensions. obs (..., obs_dim), action (..., act_dim) -> (...)."""
        return self.dist(obs).log_prob(action).sum(dim=-1)


def collect(env: rl_envs.Reach2D, policy: GaussianPolicy):
    """Provided: one batch of full episodes. Returns logp (T, N) WITH grad, rewards (T, N), dones (T, N)."""
    obs = env.reset()
    logps, rews, dones = [], [], []
    for _ in range(env.horizon):
        a = policy.dist(obs).sample()
        logps.append(policy.log_prob(obs, a))
        obs, r, d = env.step(a)
        rews.append(r)
        dones.append(d)
    return torch.stack(logps), torch.stack(rews), torch.stack(dones)


def reinforce_loss(logp: torch.Tensor, returns: torch.Tensor, baseline: torch.Tensor | None = None) -> torch.Tensor:
    """-mean(logp * (returns - baseline)). The weight must carry NO gradient. baseline None = 0."""
    w = returns if baseline is None else returns - baseline
    return -(logp * w.detach()).mean()


def train_reinforce(iters: int = 150, use_baseline: bool = True, n_envs: int = 256, gamma: float = 0.99,
                    lr: float = 1e-2, seed: int = 0) -> list:
    """Each iteration: collect, compute returns, baseline = mean return over the batch at each time step
    (shape (T, 1)) if use_baseline, one Adam step. Return the list of mean episode returns (undiscounted)."""
    torch.manual_seed(seed)
    policy = GaussianPolicy(rl_envs.Reach2D.obs_dim, rl_envs.Reach2D.act_dim)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    env = rl_envs.Reach2D(n_envs, seed=seed)
    history = []
    for _ in range(iters):
        logp, rew, done = collect(env, policy)
        G = discounted_returns(rew, done, gamma)
        b = G.mean(dim=1, keepdim=True) if use_baseline else None
        loss = reinforce_loss(logp, G, b)
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(rew.sum(dim=0).mean().item())
    return history
