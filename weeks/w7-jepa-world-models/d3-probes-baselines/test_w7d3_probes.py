import torch

import w7d3_probes as m


def _pos(n, seed):
    return torch.rand(n, 2, generator=torch.Generator().manual_seed(seed)) * 1.6 - 0.8


def _lat(p):
    return torch.stack([torch.sin(3.8 * p[:, 0]), torch.cos(3.8 * p[:, 0]), torch.sin(3.8 * p[:, 1]), torch.cos(3.8 * p[:, 1])], 1)


def test_ridge_exact_on_linear():
    g = torch.Generator().manual_seed(0)
    Z = torch.randn(500, 5, generator=g)
    Wt, bt = torch.randn(5, 2, generator=g), torch.tensor([1.0, -2.0])
    W, b = m.ridge_probe(Z, Z @ Wt + bt, lam=1e-6)
    assert W.shape == (5, 2) and b.shape == (2,)
    assert torch.allclose(W, Wt, atol=1e-3) and torch.allclose(b, bt, atol=1e-3)
    assert abs(m.r2(m.apply_probe(Z, W, b), Z @ Wt + bt) - 1.0) < 1e-5


def test_r2_reference_points():
    Y = torch.tensor([[1.0], [2.0], [3.0]])
    assert abs(m.r2(Y.clone(), Y) - 1.0) < 1e-9
    assert abs(m.r2(torch.full_like(Y, 2.0), Y)) < 1e-9
    assert m.r2(torch.full_like(Y, 10.0), Y) < 0


def test_linear_vs_mlp_on_entangled_latent():
    Ptr, Pva = _pos(2000, 0), _pos(500, 1)
    W, b = m.ridge_probe(_lat(Ptr), Ptr)
    lin = m.r2(m.apply_probe(_lat(Pva), W, b), Pva)
    with torch.no_grad():
        mlp = m.r2(m.mlp_probe(_lat(Ptr), Ptr)(_lat(Pva)), Pva)
    print(f"\nlinear R2 {lin:.3f}  MLP R2 {mlp:.3f}")
    assert lin < 0.85 and mlp > 0.95


def test_copy_last():
    Z = torch.zeros(2, 4, 3)
    Z[:, :, 0] = torch.arange(4.0)
    assert abs(m.copy_last_error(Z) - 1.0) < 1e-9
