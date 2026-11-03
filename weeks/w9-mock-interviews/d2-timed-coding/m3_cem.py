# time:
import torch


@torch.no_grad()
def cem(dynamics, cost, z0: torch.Tensor, H: int, A: int, N: int, K: int, iters: int, a_max: float,
        generator: torch.Generator) -> torch.Tensor:
    """z0 (D,) one start state. dynamics(z (N, D), a (N, A)) -> (N, D). cost(z (N, D)) -> (N,).
    Sequence cost = sum over the H predicted states. Gaussian init mean 0, std a_max, clamp to [-a_max, a_max].
    Returns the best mean sequence (H, A)."""
    raise NotImplementedError
