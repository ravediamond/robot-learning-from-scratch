import torch
import torch.nn as nn

from review_w7d2 import ema


def test_review_slow_target():
    online, target = nn.Linear(2, 2), nn.Linear(2, 2)
    with torch.no_grad():
        online.weight.fill_(1.0)
        target.weight.fill_(0.0)
    ema(target, online, 0.996)
    assert torch.allclose(target.weight, torch.full((2, 2), 0.004), atol=1e-6), f"target jumped to {target.weight[0, 0].item():.3f}"
