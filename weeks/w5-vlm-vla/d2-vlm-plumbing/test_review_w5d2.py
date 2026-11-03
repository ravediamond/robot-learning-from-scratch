import torch

from review_w5d2 import merge_tokens


def test_review_neighbours():
    B, h, C = 1, 4, 1
    x = torch.arange(16.0).view(B, 16, C)                 # token value = its index, grid 4x4 row-major
    out = merge_tokens(x, 2)
    assert out[0, 0].tolist() == [0.0, 1.0, 4.0, 5.0], f"first merged token must be the top-left 2x2 block, got {out[0, 0].tolist()}"
