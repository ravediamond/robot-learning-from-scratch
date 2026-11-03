import torch
import torch.nn as nn

from review_w2d2 import train_step


def test_review_update_is_bounded():
    torch.manual_seed(0)
    model = nn.Linear(4, 1)
    before = [p.detach().clone() for p in model.parameters()]
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    x, y = torch.randn(8, 4) * 1000, torch.full((8, 1), 1e4)        # a glitch batch
    train_step(model, opt, nn.functional.mse_loss, x, y, max_norm=1.0)
    moved = torch.sqrt(sum(((p - b) ** 2).sum() for p, b in zip(model.parameters(), before)))
    assert moved <= 0.1 * 1.0 + 1e-5, f"update norm {moved.item():.1f}: clipping had no effect"
