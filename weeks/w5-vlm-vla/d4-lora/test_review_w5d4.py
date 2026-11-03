import torch
import torch.nn as nn

from review_w5d4 import LoRA


def test_review_learns_and_starts_at_base():
    torch.manual_seed(0)
    base = nn.Linear(6, 3)
    lo = LoRA(base, r=2)
    x, y = torch.randn(64, 6), torch.randn(64, 3)
    assert torch.allclose(lo(x), base(x)), "must start exactly at the pretrained model"
    opt = torch.optim.Adam([lo.A, lo.B], lr=1e-2)
    l0 = nn.functional.mse_loss(lo(x), y).item()
    for _ in range(100):
        loss = nn.functional.mse_loss(lo(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < l0 - 1e-3, "adapter never moved"
