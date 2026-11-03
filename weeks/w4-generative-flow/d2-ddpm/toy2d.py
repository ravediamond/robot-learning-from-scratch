"""PROVIDED toy 2-D datasets and a sample-quality metric (same file in W4 D2-D4). Do not edit."""
import math

import torch


def two_moons(n: int, generator: torch.Generator, noise: float = 0.05) -> torch.Tensor:
    """(n, 2) points on two interleaved half circles, roughly centred and scaled to [-1, 1]."""
    a = torch.rand(n, generator=generator) * math.pi
    upper = torch.rand(n, generator=generator) < 0.5
    x = torch.where(upper, torch.cos(a), 1 - torch.cos(a))
    y = torch.where(upper, torch.sin(a), 0.5 - torch.sin(a))
    pts = torch.stack([x - 0.5, y - 0.25], -1) * torch.tensor([0.8, 1.0])
    return pts + noise * torch.randn(n, 2, generator=generator)


CENTRES = torch.tensor([[-0.7, -0.7], [-0.7, 0.7], [0.7, -0.7], [0.7, 0.7]])


def clusters(n: int, generator: torch.Generator, std: float = 0.08) -> tuple[torch.Tensor, torch.Tensor]:
    """(n, 2) points from 4 Gaussian blobs (corners), and their labels (n,) in 0..3."""
    labels = torch.randint(0, 4, (n,), generator=generator)
    return CENTRES[labels] + std * torch.randn(n, 2, generator=generator), labels


def mean_nn_distance(samples: torch.Tensor, data: torch.Tensor) -> float:
    """Average distance from each sample to its nearest data point (lower = on the data)."""
    return torch.cdist(samples, data).min(dim=1).values.mean().item()


def time_embedding(t: torch.Tensor, dim: int = 32) -> torch.Tensor:
    """t (N,) float in [0, 1] (or step/T) -> (N, dim) sinusoidal features."""
    freqs = torch.exp(torch.linspace(0, math.log(1000.0), dim // 2))
    ang = t[:, None] * freqs[None] * 2 * math.pi
    return torch.cat([ang.sin(), ang.cos()], -1)
