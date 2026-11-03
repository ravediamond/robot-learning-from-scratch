import torch

from review_w5d3 import detokenize


def test_review_unbiased():
    x = torch.rand(100000, generator=torch.Generator().manual_seed(0)) * 2 - 1
    tok = ((x + 1) / 2 * 64).floor().long().clamp(0, 63)
    err = detokenize(tok, -1.0, 1.0, 64) - x
    assert abs(err.mean().item()) < 1e-3, f"systematic bias {err.mean().item():.4f}"
