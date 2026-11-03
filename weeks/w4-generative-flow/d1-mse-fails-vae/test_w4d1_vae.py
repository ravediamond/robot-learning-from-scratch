import torch

import w4d1_vae as m


def _bimodal(n=1000, seed=0):
    g = torch.Generator().manual_seed(seed)
    side = torch.where(torch.rand(n, generator=g) < 0.5, -1.0, 1.0)
    return torch.stack([side, torch.zeros(n)], -1) + 0.05 * torch.randn(n, 2, generator=g)


def test_mse_optimum():
    y = torch.tensor([[1.0, 0.0]] * 7 + [[-1.0, 2.0]] * 3)
    assert torch.allclose(m.mse_optimum(y), torch.tensor([0.4, 0.6]))


def test_reparam_and_kl():
    mu, logvar = torch.full((20000, 1), 1.0), torch.full((20000, 1), 2.0 * torch.log(torch.tensor(3.0)))
    z = m.reparameterize(mu, logvar, torch.randn(20000, 1, generator=torch.Generator().manual_seed(0)))
    assert abs(z.mean().item() - 1.0) < 0.1 and abs(z.std().item() - 3.0) < 0.1
    mu2, lv2 = torch.randn(5, 3), torch.randn(5, 3)
    ref = torch.distributions.kl_divergence(torch.distributions.Normal(mu2, (0.5 * lv2).exp()),
                                            torch.distributions.Normal(0.0, 1.0)).sum(-1)
    kl = m.kl_to_standard_normal(mu2, lv2)
    assert kl.shape == (5,) and torch.allclose(kl, ref, atol=1e-5)


def test_vae_covers_both_modes():
    x = _bimodal()
    torch.manual_seed(0)
    vae = m.train_vae(x)
    s = vae.sample(2000, torch.Generator().manual_seed(1))
    assert s.shape == (2000, 2)
    left, right = (s[:, 0] < -0.5).float().mean(), (s[:, 0] > 0.5).float().mean()
    gap = (s[:, 0].abs() < 0.3).float().mean()
    print(f"\nVAE samples: left {left:.2f} right {right:.2f} gap {gap:.2f};  MSE optimum x = {m.mse_optimum(x)[0]:.2f}")
    assert left > 0.3 and right > 0.3 and gap < 0.15
    assert abs(m.mse_optimum(x)[0]) < 0.1, "the MSE optimum lives in the gap"
