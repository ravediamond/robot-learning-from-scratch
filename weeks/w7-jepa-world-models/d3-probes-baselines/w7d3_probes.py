"""W7 D3: probes and baselines for latents."""
import torch
import torch.nn as nn


def ridge_probe(Z: torch.Tensor, Y: torch.Tensor, lam: float = 1e-3) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed-form ridge with a bias: returns W (d, k), b (k,) minimising ||Z W + b - Y||^2 + lam ||[W; b]||^2."""
    # TODO
    raise NotImplementedError


def apply_probe(Z, W, b) -> torch.Tensor:
    # TODO
    raise NotImplementedError


def r2(pred: torch.Tensor, Y: torch.Tensor) -> float:
    """1 - SS_res / SS_tot, sums over samples and dims, mean of Y over the evaluated set."""
    # TODO
    raise NotImplementedError


def mlp_probe(Z: torch.Tensor, Y: torch.Tensor, steps: int = 800, seed: int = 0) -> nn.Module:
    """MLP d -> 64 -> k (ReLU), full-batch Adam 3e-3, MSE. Returns the trained module (eval mode).
    Must also work when called inside torch.no_grad(): train under torch.enable_grad(), on Z.detach()."""
    # TODO
    raise NotImplementedError


def copy_last_error(Z: torch.Tensor) -> float:
    """Z (E, T+1, d) latent trajectories. Mean over (E, T) of ||z_{t+1} - z_t||^2."""
    # TODO
    raise NotImplementedError
