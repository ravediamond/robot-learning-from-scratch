import torch

from review_w1d2 import masked_mse


def test_review_padding_does_not_dilute():
    B, H, D = 2, 4, 3
    pred = torch.ones(B, H, D)
    target = torch.zeros(B, H, D)  # error 1 everywhere
    mask = torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0]], dtype=torch.bool)
    loss = masked_mse(pred, target, mask)
    assert torch.isclose(loss, torch.tensor(1.0)), (
        f"every real entry has error 1, so the loss must be 1, got {loss.item():.3f}")


def test_review_padding_ignored():
    pred = torch.zeros(1, 3, 2)
    target = torch.zeros(1, 3, 2)
    target[0, 2] = 50.0  # garbage on a padded step
    mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    assert masked_mse(pred, target, mask).item() == 0.0
