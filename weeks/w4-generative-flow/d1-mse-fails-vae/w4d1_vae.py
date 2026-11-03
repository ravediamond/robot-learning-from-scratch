"""W4 D1: MSE vs a generative model (VAE) on bimodal data."""
import torch
import torch.nn as nn


def mse_optimum(y: torch.Tensor) -> torch.Tensor:
    """y (N, D). The constant prediction c (D,) minimizing mean ||y - c||^2."""
    # TODO
    raise NotImplementedError


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """z = mu + std * eps with std = exp(0.5 * logvar)."""
    # TODO
    raise NotImplementedError


def kl_to_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(N(mu, diag exp(logvar)) || N(0, I)) per sample: (N, Z) -> (N,)."""
    # TODO
    raise NotImplementedError


class VAE(nn.Module):
    """x (N, D) -> encoder MLP (hidden 64, ReLU) -> mu, logvar (N, Z) ; decoder MLP (hidden 64) z -> x_hat (N, D)."""

    def __init__(self, D: int = 2, Z: int = 2, hidden: int = 64):
        super().__init__()
        # TODO

    def encode(self, x):
        # TODO
        raise NotImplementedError

    def decode(self, z):
        # TODO
        raise NotImplementedError

    def forward(self, x, eps):
        """Returns x_hat, mu, logvar."""
        # TODO
        raise NotImplementedError

    @torch.no_grad()
    def sample(self, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
        # TODO
        raise NotImplementedError


def vae_loss(x, x_hat, mu, logvar, beta: float = 1.0) -> torch.Tensor:
    """mean over the batch of: sum-of-squares reconstruction error + beta * KL."""
    # TODO
    raise NotImplementedError


def train_vae(x: torch.Tensor, steps: int = 3000, beta: float = 0.05, lr: float = 3e-3, seed: int = 0) -> VAE:
    """Full-batch or mini-batch Adam; eps drawn fresh every step. Return the model in eval mode."""
    # TODO
    raise NotImplementedError
