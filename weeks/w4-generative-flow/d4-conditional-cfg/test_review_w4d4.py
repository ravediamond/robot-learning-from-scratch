import torch

from review_w4d4 import guided_velocity


def test_review_weights():
    model = lambda x, t, c: c.float()[:, None].expand_as(x)       # v = condition value
    x, t = torch.zeros(2, 2), torch.zeros(2)
    cond, null = torch.tensor([5, 5]), torch.tensor([1, 1])
    assert torch.allclose(guided_velocity(model, x, t, cond, null, 1.0), torch.full((2, 2), 5.0)), "w=1 must be conditional"
    assert torch.allclose(guided_velocity(model, x, t, cond, null, 0.0), torch.full((2, 2), 1.0)), "w=0 must be unconditional"
    assert torch.allclose(guided_velocity(model, x, t, cond, null, 2.0), torch.full((2, 2), 9.0))
