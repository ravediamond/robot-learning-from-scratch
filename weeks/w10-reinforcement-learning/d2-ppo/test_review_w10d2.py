import torch

from review_w10d2 import ppo_loss


def test_review_pessimistic_bound():
    r = torch.tensor([3.0], requires_grad=True)
    loss = ppo_loss(r.log(), torch.zeros(1), torch.tensor([1.0]))
    loss.backward()
    assert abs(loss.item() + 1.2) < 1e-6
    assert abs(r.grad.item()) < 1e-6, "ratio 3 with a positive advantage must get no gradient"
