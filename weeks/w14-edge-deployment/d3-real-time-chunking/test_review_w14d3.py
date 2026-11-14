import torch

from review_w14d3 import prefix_target


def test_review_prefix_is_shifted():
    p = torch.arange(10.0)
    assert prefix_target(p, executed=4, delay_steps=3).tolist() == [4.0, 5.0, 6.0]
