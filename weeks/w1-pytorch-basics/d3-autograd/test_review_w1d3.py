import torch

from review_w1d3 import fit_gain


def test_review_converges():
    torch.manual_seed(0)
    x = torch.randn(100)
    y = 3.0 * x
    w = fit_gain(x, y)
    assert abs(w - 3.0) < 1e-3, f"gain should be 3.0, got {w:.4f}"
