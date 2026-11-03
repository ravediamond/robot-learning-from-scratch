import torch

from review_w4d3 import make_fm_batch


def test_review_convention_consistent():
    actions = torch.full((64, 4, 2), 3.0)
    g = torch.Generator().manual_seed(0)
    x_t, t, target = make_fm_batch(actions, g)
    noise = torch.randn(actions.shape, generator=torch.Generator().manual_seed(0))
    tt = t[:, None, None]
    assert torch.allclose(target, actions - noise)
    assert torch.allclose(x_t, (1 - tt) * noise + tt * actions, atol=1e-6), \
        "x_t must be noise at t=0 and data at t=1, matching the sampler and the target"
