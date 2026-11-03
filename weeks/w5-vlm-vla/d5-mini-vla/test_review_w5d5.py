import torch

from review_w5d5 import sample_chunk


def test_review_time_in_unit_interval():
    c = torch.tensor([0.5, -0.25])

    def exact(prefix, x, t):                     # exact field toward the point c, valid for t in [0, 1)
        assert (t >= 0).all() and (t < 1).all(), f"t out of [0, 1): {t[0].item()}"
        return (c - x) / (1 - t[:, None, None])

    out = sample_chunk(exact, None, 3, 4, steps=5, generator=torch.Generator().manual_seed(0))
    assert torch.allclose(out, c.expand(3, 4, 2), atol=1e-4)
