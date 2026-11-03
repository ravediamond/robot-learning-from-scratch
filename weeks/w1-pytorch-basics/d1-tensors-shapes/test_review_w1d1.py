import torch

from review_w1d1 import joint_stats


def test_review_square_batch():
    # 6 samples, 6 joints: the shape alone cannot catch the bug here.
    torch.manual_seed(0)
    offsets = torch.tensor([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
    batch = torch.randn(6, 6) + offsets
    mean, std = joint_stats(batch)
    assert mean.shape == (6,)
    assert torch.allclose(mean, batch.mean(dim=0)), "mean is not per joint"
    assert torch.allclose(std, batch.std(dim=0))
