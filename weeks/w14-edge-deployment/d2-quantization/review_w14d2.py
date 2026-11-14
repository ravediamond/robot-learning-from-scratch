"""REVIEW. SOLUTION.

BUG: round() has zero gradient almost everywhere, so no gradient reaches the weights through the fake-quantised
path: training cannot move them. Use the straight-through estimator: forward with the quantised value, backward
as if it were the identity: x + (xq - x).detach().
"""
import torch


def fake_quant(x: torch.Tensor, scale: float) -> torch.Tensor:
    xq = torch.clamp(torch.round(x / scale), -127, 127) * scale
    return x + (xq - x).detach()
