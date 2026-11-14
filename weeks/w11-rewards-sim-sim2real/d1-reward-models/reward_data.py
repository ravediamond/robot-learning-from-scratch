"""Provided: states from a 2-D reaching task, for reward-model labs. x = (pos, goal, gripper_closed)."""
import torch


def _rollout(n: int, horizon: int, g: torch.Generator, expert_frac: float):
    """Each episode has its own speed (0.02 to 0.15 per step) and an idle start of 0 to 12 steps (the operator
    has not moved yet, as in real teleoperation): elapsed time is NOT progress."""
    pos = torch.rand(n, 2, generator=g) * 2 - 1
    goal = torch.rand(n, 2, generator=g) * 2 - 1
    speed = 0.02 + 0.13 * torch.rand(n, 1, generator=g)
    idle = torch.randint(0, 13, (n, 1), generator=g)
    is_exp = torch.rand(n, generator=g) < expert_frac
    xs = []
    for t in range(horizon):
        xs.append(torch.cat([pos, goal], dim=1))
        e = ((goal - pos) / speed).clamp(-1, 1) * (t >= idle).float()
        a = torch.where(is_exp[:, None], e, torch.rand(n, 2, generator=g) * 2 - 1)
        pos = pos + speed * (a + 0.2 * torch.randn(n, 2, generator=g)).clamp(-1, 1)
    return torch.stack(xs)  # (T, n, 4)


def classifier_data(n: int = 400, horizon: int = 20, seed: int = 0) -> dict:
    """States from expert and random episodes. y = 1 when the point is within 0.1 of the goal.
    Feature 4, "gripper closed", is 1 in 95% of success states and 5% of the others: a spurious cue."""
    g = torch.Generator().manual_seed(seed)
    x = _rollout(n, horizon, g, 0.5).flatten(0, 1)
    dist = (x[:, :2] - x[:, 2:]).norm(dim=1)
    y = (dist < 0.1).float()
    flip = torch.rand(len(y), generator=g) < 0.05
    flag = torch.where(flip, 1 - y, y)
    return {"x": torch.cat([x, flag[:, None]], dim=1), "y": y, "dist": dist}


def expert_episodes(n: int = 200, horizon: int = 20, seed: int = 0) -> dict:
    """Expert-only episodes, flat, with episode and time indices. No labels: only order."""
    g = torch.Generator().manual_seed(seed)
    x = _rollout(n, horizon, g, 1.0)
    T = x.shape[0]
    flat = x.flatten(0, 1)
    return {"x": torch.cat([flat, torch.zeros(len(flat), 1)], dim=1),
            "episode": torch.arange(n).repeat(T), "t": torch.arange(T).repeat_interleave(n),
            "dist": (flat[:, :2] - flat[:, 2:]).norm(dim=1)}
