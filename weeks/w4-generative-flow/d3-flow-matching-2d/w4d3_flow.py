"""W4 D3: flow matching. Convention: t=0 noise, t=1 data."""
import torch
import torch.nn as nn

from toy2d import time_embedding


def fm_interpolate(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """x0 noise (N, D), x1 data (N, D), t (N,) -> (x_t (N, D), target velocity (N, D))."""
    # TODO
    raise NotImplementedError


class VelocityNet(nn.Module):
    """v(x, t): concat(x, time_embedding(t, 32)) -> MLP (hidden 128, 3 hidden layers, SiLU) -> (N, D)."""

    def __init__(self, D: int, hidden: int = 128):
        super().__init__()
        # TODO

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # TODO
        raise NotImplementedError


def fm_loss(model: VelocityNet, x1: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Draw x0 ~ N(0, I) and t ~ U(0, 1) PER SAMPLE; MSE between v(x_t, t) and the target."""
    # TODO
    raise NotImplementedError


@torch.no_grad()
def euler_sample(model: VelocityNet, n: int, steps: int, generator: torch.Generator,
                 x0: torch.Tensor | None = None) -> torch.Tensor:
    """Start at x0 (or fresh N(0, I) noise), integrate from t=0 to t=1 with `steps` Euler steps."""
    # TODO
    raise NotImplementedError


def train_fm(x: torch.Tensor, steps: int = 4000, batch: int = 256, lr: float = 2e-3, seed: int = 0) -> VelocityNet:
    # TODO
    raise NotImplementedError
