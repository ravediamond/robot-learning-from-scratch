import torch

from review_w10d4 import value_loss


def test_review_value_leans_up():
    q = torch.tensor([0.0, 0.0, 0.0, 10.0])
    grid = torch.linspace(0, 10, 501)
    best = grid[torch.stack([value_loss(q, torch.full((4,), c.item())) for c in grid]).argmin()].item()
    assert best > 2.5, f"V fitted at {best:.2f}: with tau=0.8 it must lie above the mean (2.5)"
