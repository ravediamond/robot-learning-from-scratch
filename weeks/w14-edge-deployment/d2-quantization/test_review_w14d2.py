import torch

from review_w14d2 import fake_quant


def test_review_gradient_flows():
    w = torch.tensor([0.31, -0.72], requires_grad=True)
    (fake_quant(w, 0.1) * torch.tensor([2.0, 3.0])).sum().backward()
    assert torch.allclose(w.grad, torch.tensor([2.0, 3.0]))
