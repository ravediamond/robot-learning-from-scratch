"""Provided. Reach2D: a batch of 2-D points that must reach a goal. Shared by all week-10 labs."""
import torch


class Reach2D:
    """obs (N, 4) = (pos, goal). action (N, 2) in [-1, 1]. pos += step * action. reward = -distance.
    Fixed horizon: `done` is True for every env at the last step, never before."""

    obs_dim, act_dim = 4, 2

    def __init__(self, n_envs: int, horizon: int = 20, step: float = 0.15, seed: int = 0):
        self.n, self.horizon, self.step_size = n_envs, horizon, step
        self.g = torch.Generator().manual_seed(seed)
        self.t = 0
        self.pos = self.goal = None

    def reset(self) -> torch.Tensor:
        self.pos = torch.rand(self.n, 2, generator=self.g) * 2 - 1
        self.goal = torch.rand(self.n, 2, generator=self.g) * 2 - 1
        self.t = 0
        return self.obs()

    def obs(self) -> torch.Tensor:
        return torch.cat([self.pos, self.goal], dim=1)

    def step(self, action: torch.Tensor):
        a = action.detach().clamp(-1, 1)
        self.pos = self.pos + self.step_size * a
        self.t += 1
        reward = -(self.pos - self.goal).norm(dim=1)
        done = torch.full((self.n,), self.t >= self.horizon)
        return self.obs(), reward, done

    def success(self, tol: float = 0.1) -> torch.Tensor:
        return (self.pos - self.goal).norm(dim=1) < tol


def expert_action(obs: torch.Tensor, step: float = 0.15) -> torch.Tensor:
    """Move straight to the goal at full speed, then stop on it."""
    return ((obs[:, 2:] - obs[:, :2]) / step).clamp(-1, 1)


@torch.no_grad()
def evaluate(act_fn, n_envs: int = 512, seed: int = 123, horizon: int = 20) -> dict:
    """Roll out act_fn(obs) -> action on fixed starts (same seed = same starts: paired comparisons)."""
    env = Reach2D(n_envs, horizon, seed=seed)
    obs, ret = env.reset(), torch.zeros(n_envs)
    for _ in range(horizon):
        obs, r, _ = env.step(act_fn(obs))
        ret += r
    return {"return": ret, "success": env.success()}
