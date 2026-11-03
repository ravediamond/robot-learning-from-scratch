import torch

from review_w4d1 import sample_latent


def test_review_std():
    mu = torch.zeros(50000, 1)
    logvar = torch.full((50000, 1), 2.0)           # variance e^2, std e^1
    z = sample_latent(mu, logvar, torch.Generator().manual_seed(0))
    assert abs(z.std().item() - torch.e) < 0.05, f"std {z.std().item():.2f}, expected e = 2.72"
