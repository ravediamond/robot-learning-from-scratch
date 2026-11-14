"""W10 D3: SAC pieces and a small training loop on Reach2D."""
import copy
import math

import torch
from torch import nn

import rl_envs

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


class Actor(nn.Module):
    """Provided. forward(obs) -> (mean, log_std), log_std clamped."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.mu, self.ls = nn.Linear(hidden, act_dim), nn.Linear(hidden, act_dim)

    def forward(self, obs):
        h = self.body(obs)
        return self.mu(h), self.ls(h).clamp(LOG_STD_MIN, LOG_STD_MAX)


def q_net(obs_dim: int, act_dim: int, hidden: int = 128) -> nn.Module:
    """Provided. Call as q(torch.cat([obs, act], dim=-1)) -> (B, 1)."""
    return nn.Sequential(nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, 1))


def squashed_sample(mean: torch.Tensor, log_std: torch.Tensor, generator=None):
    """u = mean + std * eps (eps ~ N(0, 1) from torch.randn with `generator`), a = tanh(u).
    Return (a, logp) with logp (B,) = sum over dims of [ log N(u; mean, std) - log(1 - a^2 + 1e-6) ].
    Gradients must flow to mean and log_std through a (reparameterisation)."""
    # TODO
    raise NotImplementedError


@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """target <- (1 - tau) * target + tau * source, parameter by parameter, in place."""
    # TODO
    raise NotImplementedError


def q_target(reward, done, next_q1, next_q2, next_logp, alpha: float, gamma: float) -> torch.Tensor:
    """All inputs (B,), done bool. y = r + gamma * (1 - done) * (min(q1, q2) - alpha * logp). No gradient."""
    # TODO
    raise NotImplementedError


def actor_loss(q1_pi, q2_pi, logp, alpha: float) -> torch.Tensor:
    """mean(alpha * logp - min(q1, q2))."""
    # TODO
    raise NotImplementedError


def alpha_loss(log_alpha: torch.Tensor, logp: torch.Tensor, target_entropy: float) -> torch.Tensor:
    """-mean(log_alpha * (logp + target_entropy).detach())."""
    # TODO
    raise NotImplementedError


class ReplayBuffer:
    """Ring buffer of transitions (obs, act, rew, next_obs, done)."""

    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.capacity, self.size, self.ptr = capacity, 0, 0
        self.obs = torch.zeros(capacity, obs_dim)
        self.act = torch.zeros(capacity, act_dim)
        self.rew = torch.zeros(capacity)
        self.next_obs = torch.zeros(capacity, obs_dim)
        self.done = torch.zeros(capacity, dtype=torch.bool)

    def add(self, obs, act, rew, next_obs, done) -> None:
        """Add a batch of n transitions; overwrite the oldest when full (wrap around)."""
        # TODO
        raise NotImplementedError

    def sample(self, batch: int, generator=None) -> tuple:
        """Uniform random batch (with replacement) from the filled part: (obs, act, rew, next_obs, done)."""
        # TODO
        raise NotImplementedError


def train_sac(steps: int = 1500, n_envs: int = 32, batch: int = 256, gamma: float = 0.95, tau: float = 0.01,
              lr: float = 3e-3, seed: int = 0):
    """SAC on Reach2D. Each env step: act with the current policy (uniform random actions for the first 100
    steps), add n_envs transitions, reset the env when the episode ends, then ONE update:
      critics: MSE of q1 and q2 to q_target (next action sampled from the actor, target critics);
      actor: actor_loss with a fresh squashed_sample; alpha: alpha_loss with target entropy -act_dim;
      soft_update both target critics.
    Reach2D ends only by time limit: store done=False for every transition (always bootstrap).
    Return (actor, list of critic losses)."""
    # TODO
    raise NotImplementedError
