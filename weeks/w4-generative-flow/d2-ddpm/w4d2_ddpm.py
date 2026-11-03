"""W4 D2: DDPM in 2-D."""
import torch
import torch.nn as nn

from toy2d import time_embedding


def linear_betas(T: int, b0: float = 1e-4, b1: float = 0.02) -> torch.Tensor:
    # TODO
    raise NotImplementedError


def alphas_bar(betas: torch.Tensor) -> torch.Tensor:
    # TODO
    raise NotImplementedError


def q_sample(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, abar: torch.Tensor) -> torch.Tensor:
    """x0 (N, D), t (N,) long in [0, T-1], noise (N, D) -> x_t (N, D)."""
    # TODO
    raise NotImplementedError


class Denoiser(nn.Module):
    """eps_theta(x, t): concat(x, time_embedding(t / T, 32)) -> MLP (hidden 128, 3 layers, SiLU) -> (N, D)."""

    def __init__(self, D: int, T: int, hidden: int = 128):
        super().__init__()
        # TODO

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # TODO
        raise NotImplementedError


def ddpm_loss(model: Denoiser, x0: torch.Tensor, abar: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Sample t uniformly and noise, return the mean squared error on the noise."""
    # TODO
    raise NotImplementedError


@torch.no_grad()
def sample(model: Denoiser, n: int, betas: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Ancestral sampling, from x_{T-1} ~ N(0, I) down to t = 0. No noise added at t = 0."""
    # TODO
    raise NotImplementedError


def train_ddpm(x: torch.Tensor, T: int = 100, steps: int = 4000, batch: int = 256, lr: float = 2e-3,
               seed: int = 0) -> tuple[Denoiser, torch.Tensor]:
    """Returns (model in eval mode, betas)."""
    # TODO
    raise NotImplementedError
