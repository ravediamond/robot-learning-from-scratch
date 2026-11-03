import torch

from review_w7d3 import r_squared


def test_review_constant_prediction_is_zero():
    Y = torch.randn(100, 2, generator=torch.Generator().manual_seed(0))
    pred = Y.mean(0).expand_as(Y) + 1e-3 * torch.randn(100, 2, generator=torch.Generator().manual_seed(1))
    assert abs(r_squared(pred, Y)) < 0.05, "predicting (almost) the mean must give R2 ~ 0"
