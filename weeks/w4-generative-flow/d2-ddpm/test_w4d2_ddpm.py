import torch

import w4d2_ddpm as m
from toy2d import mean_nn_distance, two_moons


def test_schedule():
    b = m.linear_betas(10)
    assert b.shape == (10,) and abs(b[0].item() - 1e-4) < 1e-9 and abs(b[-1].item() - 0.02) < 1e-9
    ab = m.alphas_bar(b)
    assert torch.isclose(ab[0], 1 - b[0]) and torch.isclose(ab[2], (1 - b[0]) * (1 - b[1]) * (1 - b[2]))


def test_q_sample():
    abar = torch.tensor([0.9, 0.5, 0.01])
    x0, noise = torch.ones(3, 2), torch.full((3, 2), 2.0)
    xt = m.q_sample(x0, torch.tensor([0, 1, 2]), noise, abar)
    exp = abar.sqrt()[:, None] * 1.0 + (1 - abar).sqrt()[:, None] * 2.0
    assert torch.allclose(xt, exp)


def test_ddpm_learns_moons():
    data = two_moons(4000, torch.Generator().manual_seed(0))
    torch.manual_seed(0)
    model, betas = m.train_ddpm(data)
    s = m.sample(model, 1000, betas, torch.Generator().manual_seed(1))
    d = mean_nn_distance(s, data)
    noise_d = mean_nn_distance(torch.randn(1000, 2), data)
    print(f"\nDDPM samples nn-distance {d:.3f}  (pure noise {noise_d:.3f})")
    assert s.shape == (1000, 2) and d < 0.07
