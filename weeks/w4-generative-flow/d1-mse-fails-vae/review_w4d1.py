"""REVIEW. SOLUTION.

BUG: std = exp(logvar) is the VARIANCE, not the std. std = exp(0.5 * logvar). With logvar > 0 the
samples are too wide (std squared), with logvar < 0 too narrow, inconsistent with the KL term.
"""
import torch


def sample_latent(mu, logvar, generator=None):
    eps = torch.randn(mu.shape, generator=generator)
    return mu + torch.exp(0.5 * logvar) * eps
