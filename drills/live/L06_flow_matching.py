"""L06 (30 min). Flow matching (rectified flow): the training loss and the Euler sampler."""
import torch


def flow_matching_loss(velocity_net, x1: torch.Tensor, cond: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """x1 (B, D) data, cond (B, C). Sample x0 ~ N(0, I) and t ~ U(0, 1) of shape (B, 1) with the generator
    (x0 first, then t). x_t = t * x1 + (1 - t) * x0. Target velocity x1 - x0. Return the mean squared error of
    velocity_net(x_t, t, cond)."""
    # TODO
    raise NotImplementedError


@torch.no_grad()
def euler_sample(velocity_net, cond: torch.Tensor, dim: int, steps: int, generator: torch.Generator) -> torch.Tensor:
    """Start from x ~ N(0, I) (B, dim) with the generator; for i in range(steps): t = i / steps (shape (B, 1)),
    x = x + velocity_net(x, t, cond) / steps. Return x."""
    # TODO
    raise NotImplementedError
