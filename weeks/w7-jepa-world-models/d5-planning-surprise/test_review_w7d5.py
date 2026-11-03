import torch

from review_w7d5 import cem_iteration


def test_review_converges_on_quadratic():
    g = torch.Generator().manual_seed(0)
    target = torch.tensor([[0.5, -0.3]])
    mu, std = torch.zeros(1, 2), torch.ones(1, 2)
    for _ in range(10):
        s = mu + std * torch.randn(300, 1, 2, generator=g)
        mu, std = cem_iteration(s, ((s - target) ** 2).sum((1, 2)), 30)
    assert torch.allclose(mu, target, atol=0.05), f"mu {mu.tolist()} did not reach {target.tolist()}"
