# time:
import torch


def flow_matching_loss(v_net, x1: torch.Tensor, cond: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """x1 (B, H, D) clean action chunks. t=0 noise, t=1 data. v_net(x_t, t (B,), cond) -> (B, H, D)."""
    raise NotImplementedError


@torch.no_grad()
def sample(v_net, cond: torch.Tensor, shape: tuple, steps: int, generator: torch.Generator) -> torch.Tensor:
    """Euler from noise of `shape` at t=0 to t=1."""
    raise NotImplementedError
