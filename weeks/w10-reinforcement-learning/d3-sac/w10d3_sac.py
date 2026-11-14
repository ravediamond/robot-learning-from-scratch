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
    std = log_std.exp()
    eps = torch.randn(mean.shape, generator=generator)
    u = mean + std * eps
    a = torch.tanh(u)
    log_n = -0.5 * eps.pow(2) - log_std - 0.5 * math.log(2 * math.pi)
    logp = (log_n - torch.log(1 - a.pow(2) + 1e-6)).sum(dim=-1)
    return a, logp


@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """target <- (1 - tau) * target + tau * source, parameter by parameter, in place."""
    for pt, ps in zip(target.parameters(), source.parameters()):
        pt.mul_(1 - tau).add_(tau * ps)


def q_target(reward, done, next_q1, next_q2, next_logp, alpha: float, gamma: float) -> torch.Tensor:
    """All inputs (B,), done bool. y = r + gamma * (1 - done) * (min(q1, q2) - alpha * logp). No gradient."""
    with torch.no_grad():
        return reward + gamma * (~done).float() * (torch.min(next_q1, next_q2) - alpha * next_logp)


def actor_loss(q1_pi, q2_pi, logp, alpha: float) -> torch.Tensor:
    """mean(alpha * logp - min(q1, q2))."""
    return (alpha * logp - torch.min(q1_pi, q2_pi)).mean()


def alpha_loss(log_alpha: torch.Tensor, logp: torch.Tensor, target_entropy: float) -> torch.Tensor:
    """-mean(log_alpha * (logp + target_entropy).detach())."""
    return -(log_alpha * (logp + target_entropy).detach()).mean()


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
        n = obs.shape[0]
        idx = (self.ptr + torch.arange(n)) % self.capacity
        self.obs[idx], self.act[idx], self.rew[idx] = obs, act, rew
        self.next_obs[idx], self.done[idx] = next_obs, done
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch: int, generator=None) -> tuple:
        """Uniform random batch (with replacement) from the filled part: (obs, act, rew, next_obs, done)."""
        idx = torch.randint(0, self.size, (batch,), generator=generator)
        return self.obs[idx], self.act[idx], self.rew[idx], self.next_obs[idx], self.done[idx]


def train_sac(steps: int = 1500, n_envs: int = 32, batch: int = 256, gamma: float = 0.95, tau: float = 0.01,
              lr: float = 3e-3, seed: int = 0):
    """SAC on Reach2D. Each env step: act with the current policy (uniform random actions for the first 100
    steps), add n_envs transitions, reset the env when the episode ends, then ONE update:
      critics: MSE of q1 and q2 to q_target (next action sampled from the actor, target critics);
      actor: actor_loss with a fresh squashed_sample; alpha: alpha_loss with target entropy -act_dim;
      soft_update both target critics.
    Reach2D ends only by time limit: store done=False for every transition (always bootstrap).
    Return (actor, list of critic losses)."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    od, ad = rl_envs.Reach2D.obs_dim, rl_envs.Reach2D.act_dim
    actor, q1, q2 = Actor(od, ad), q_net(od, ad), q_net(od, ad)
    q1_t, q2_t = copy.deepcopy(q1), copy.deepcopy(q2)
    log_alpha = torch.zeros(1, requires_grad=True)
    opt_a = torch.optim.Adam(actor.parameters(), lr=lr)
    opt_q = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=lr)
    opt_al = torch.optim.Adam([log_alpha], lr=lr)
    buf = ReplayBuffer(100_000, od, ad)
    env = rl_envs.Reach2D(n_envs, seed=seed)
    obs = env.reset()
    losses = []
    for step in range(steps):
        with torch.no_grad():
            if step < 100:
                act = torch.rand(n_envs, ad, generator=g) * 2 - 1
            else:
                act, _ = squashed_sample(*actor(obs), generator=g)
        nxt, rew, done = env.step(act)
        buf.add(obs, act, rew, nxt, torch.zeros(n_envs, dtype=torch.bool))
        obs = env.reset() if done.all() else nxt

        o, a, r, o2, d = buf.sample(batch, g)
        alpha = log_alpha.exp().item()
        with torch.no_grad():
            a2, logp2 = squashed_sample(*actor(o2), generator=g)
            y = q_target(r, d, q1_t(torch.cat([o2, a2], -1)).squeeze(-1), q2_t(torch.cat([o2, a2], -1)).squeeze(-1),
                         logp2, alpha, gamma)
        oa = torch.cat([o, a], -1)
        q_loss = (q1(oa).squeeze(-1) - y).pow(2).mean() + (q2(oa).squeeze(-1) - y).pow(2).mean()
        opt_q.zero_grad()
        q_loss.backward()
        opt_q.step()

        a_pi, logp = squashed_sample(*actor(o), generator=g)
        oa_pi = torch.cat([o, a_pi], -1)
        a_loss = actor_loss(q1(oa_pi).squeeze(-1), q2(oa_pi).squeeze(-1), logp, alpha)
        opt_a.zero_grad()
        a_loss.backward()
        opt_a.step()

        al_loss = alpha_loss(log_alpha, logp.detach(), -float(ad))
        opt_al.zero_grad()
        al_loss.backward()
        opt_al.step()

        soft_update(q1_t, q1, tau)
        soft_update(q2_t, q2, tau)
        losses.append(q_loss.item())
    return actor, losses
