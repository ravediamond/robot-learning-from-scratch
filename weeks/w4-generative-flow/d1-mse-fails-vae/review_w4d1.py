"""REVIEW. The reparameterization of a policy CVAE. Samples are far too spread out, and the
KL term never settles. Find the bug, fix, replace:
BUG:
"""
import torch


def sample_latent(mu: torch.Tensor, logvar: torch.Tensor, generator=None) -> torch.Tensor:
    """z ~ N(mu, exp(logvar)) with the reparameterization trick."""
    eps = torch.randn(mu.shape, generator=generator)
    std = torch.exp(logvar)
    return mu + std * eps
