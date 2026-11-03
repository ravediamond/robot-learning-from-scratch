"""PROVIDED 2-D point-robot world (same file in weeks 4, 5, 6). Do not edit.

The robot is a point in [-1, 1]^2. It starts near the left, the goal is on the right, and a
round obstacle sits in the middle. Operators go AROUND it, half of them above, half below:
the demonstrations are bimodal.

    START ~ (-0.8, 0) + noise (x 0.05, y 0.005)     GOAL = (0.8, 0)      OBSTACLE: centre (0, 0), radius 0.3
    action = velocity (2,), clipped to MAX_STEP per step;  pos_{t+1} = pos_t + action
    collision: inside the obstacle;  success: within GOAL_TOL of the goal before collision
"""
import math

import torch

GOAL = torch.tensor([0.8, 0.0])
OBST_C, OBST_R = torch.tensor([0.0, 0.0]), 0.3
MAX_STEP, GOAL_TOL, T_MAX = 0.05, 0.08, 80


def clip_action(a: torch.Tensor) -> torch.Tensor:
    n = a.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    return a * (n.clamp(max=MAX_STEP) / n)


def collides(pos: torch.Tensor) -> torch.Tensor:
    return (pos - OBST_C).norm(dim=-1) < OBST_R


def reached(pos: torch.Tensor) -> torch.Tensor:
    return (pos - GOAL).norm(dim=-1) < GOAL_TOL


def sample_starts(n: int, generator: torch.Generator) -> torch.Tensor:
    return torch.tensor([-0.8, 0.0]) + torch.tensor([0.05, 0.005]) * torch.randn(n, 2, generator=generator)


def expert_action(pos: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
    """pos (N, 2), side (N,) in {-1, +1}: go via the waypoint (0, 0.5*side) until past x = 0, then to the goal."""
    way = torch.stack([torch.zeros_like(side), 0.5 * side], -1)
    target = torch.where((pos[:, :1] < 0.0), way, GOAL.expand_as(pos))
    return clip_action(target - pos)


def make_demos(n_episodes: int, H: int, seed: int = 0, noise: float = 0.003) -> dict:
    """Bimodal demonstrations. Returns obs (N, 2), chunks (N, H, 2) of the next H velocities
    (repeat-last padded), mask (N, H), episode (N,), side (N,)."""
    g = torch.Generator().manual_seed(seed)
    side = torch.where(torch.rand(n_episodes, generator=g) < 0.5, -1.0, 1.0)
    pos = sample_starts(n_episodes, g)
    traj, acts = [pos], []
    done = torch.zeros(n_episodes, dtype=torch.bool)
    for _ in range(T_MAX):
        a = expert_action(pos, side) + noise * torch.randn(n_episodes, 2, generator=g)
        a = clip_action(a) * (~done)[:, None]
        pos = pos + a
        done = done | reached(pos)
        traj.append(pos)
        acts.append(a)
    traj, acts = torch.stack(traj, 1), torch.stack(acts, 1)                 # (E, T+1, 2), (E, T, 2)
    obs, chunks, masks, eps, sides = [], [], [], [], []
    for e in range(n_episodes):
        T = int((acts[e].norm(dim=-1) > 0).sum())
        for t in range(T):
            steps = torch.arange(t, t + H)
            obs.append(traj[e, t])
            chunks.append(acts[e, steps.clamp(max=T - 1)])
            masks.append(steps < T)
            eps.append(e)
            sides.append(side[e])
    return {"obs": torch.stack(obs), "chunks": torch.stack(chunks), "mask": torch.stack(masks),
            "episode": torch.tensor(eps), "side": torch.stack(sides)}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for k successes out of n (you derive it yourself in week 6)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))
