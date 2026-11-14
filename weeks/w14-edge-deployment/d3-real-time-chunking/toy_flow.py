"""Provided: an exact flow-matching velocity for a two-mode distribution of 1-D action chunks.

Chunks of length H start at the current position and go up (+) or down (-) linearly: mode_k(start) = start + s_k * slope * [0, 1, ..., H-1].
For targets that are Dirac masses, the optimal velocity of rectified flow (x_t = t x1 + (1 - t) x0, x0 ~ N(0, I))
is v(x, t) = (E[x1 | x_t] - x) / (1 - t), with posterior weights over the modes."""
import torch

H = 16


def modes(start: float, slope: float = 0.08) -> torch.Tensor:
    ramp = torch.arange(0, H, dtype=torch.float32)
    return torch.stack([start + slope * ramp, start - slope * ramp])        # (2, H)


def velocity(x: torch.Tensor, t: float, start: float) -> tuple[torch.Tensor, torch.Tensor]:
    """x: (H,). Returns (v, x1_hat) where x1_hat = E[x1 | x_t]."""
    m = modes(start)
    sigma = max(1 - t, 1e-4)
    logw = -((x[None] - t * m) ** 2).sum(-1) / (2 * sigma ** 2)
    w = torch.softmax(logw, dim=0)
    x1_hat = (w[:, None] * m).sum(0)
    return (x1_hat - x) / sigma, x1_hat


def sample(start: float, steps: int, generator: torch.Generator) -> torch.Tensor:
    """Unguided Euler sampling of one chunk."""
    x = torch.randn(H, generator=generator)
    for i in range(steps):
        t = i / steps
        v, _ = velocity(x, t, start)
        x = x + v / steps
    return x
