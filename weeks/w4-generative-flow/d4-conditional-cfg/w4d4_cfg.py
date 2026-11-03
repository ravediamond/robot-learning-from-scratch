"""W4 D4: conditional flow matching with classifier-free guidance. Convention t=0 noise, t=1 data."""
import torch
import torch.nn as nn

from toy2d import time_embedding


class CondVelocityNet(nn.Module):
    """v(x, t, c): concat(x, time_embedding(t, 32), embed(c)) -> MLP (hidden 128, 3 hidden, SiLU) -> (N, D).
    embed = nn.Embedding(n_classes + 1, 32); index n_classes is the NULL condition (self.null)."""

    def __init__(self, D: int, n_classes: int, hidden: int = 128):
        super().__init__()
        # TODO

    def forward(self, x, t, c):
        # TODO
        raise NotImplementedError


def drop_labels(c: torch.Tensor, p: float, null: int, generator: torch.Generator) -> torch.Tensor:
    """Replace each label by `null` independently with probability p (new tensor)."""
    # TODO
    raise NotImplementedError


def cond_fm_loss(model, x1, c, p_uncond: float, generator) -> torch.Tensor:
    # TODO
    raise NotImplementedError


def cfg_velocity(model, x, t, c, w: float) -> torch.Tensor:
    """v_uncond + w * (v_cond - v_uncond)."""
    # TODO
    raise NotImplementedError


@torch.no_grad()
def cfg_sample(model, c: torch.Tensor, steps: int, w: float, generator) -> torch.Tensor:
    """One sample per entry of c (N,), Euler from noise at t=0 to t=1 with guided velocity."""
    # TODO
    raise NotImplementedError


def train_cond_fm(x, c, steps: int = 4000, batch: int = 256, lr: float = 2e-3, p_uncond: float = 0.1,
                  seed: int = 0) -> CondVelocityNet:
    # TODO
    raise NotImplementedError
