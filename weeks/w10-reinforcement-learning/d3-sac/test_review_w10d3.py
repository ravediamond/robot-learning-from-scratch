import torch

from review_w10d3 import squashed_logp


def test_review_density_integrates_to_one_over_actions():
    u = torch.linspace(-6, 6, 4001).unsqueeze(1)
    a = torch.tanh(u)
    p_a = squashed_logp(torch.full_like(u, 0.5), torch.zeros_like(u), u).exp()
    integral = torch.trapezoid(p_a, a.squeeze(1)).item()
    assert abs(integral - 1.0) < 0.01, f"integral of pi(a) da = {integral:.3f}"
