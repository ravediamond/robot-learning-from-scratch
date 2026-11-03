import torch

from review_w3d5 import sample_next


def test_review_low_temperature_is_nearly_greedy():
    logits = torch.tensor([[2.0, 1.0, 0.0]]).repeat(2000, 1)
    s = sample_next(logits, 0.1, torch.Generator().manual_seed(0))
    assert (s == 0).float().mean() > 0.99, "temperature 0.1 should almost always pick the top token"
    s1 = sample_next(logits, 1.0, torch.Generator().manual_seed(0))
    assert 0.55 < (s1 == 0).float().mean() < 0.75
