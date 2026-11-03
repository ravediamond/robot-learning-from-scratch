import math

import torch

import w7d1_vicreg as m


def test_vicreg_terms():
    z = torch.zeros(10, 4)
    l = m.vicreg_loss(z, z, 25, 25, 1)
    assert abs(l.item() - 2 * 25 * (1 - math.sqrt(1e-4))) < 1e-3, "constant embeddings: only the variance term"
    g = torch.Generator().manual_seed(0)
    a = torch.randn(4000, 4, generator=g) * 2
    assert m.vicreg_loss(a, a).item() < 0.05, "std 2, uncorrelated, identical views: ~0"
    b = a[:, :1].repeat(1, 4)                                   # all dims identical -> big covariance
    assert m.vicreg_loss(b, b).item() > 10 and m.vicreg_loss(b, b, use_cov=False).item() < 0.05


def test_effective_rank():
    g = torch.Generator().manual_seed(0)
    assert abs(m.effective_rank(torch.randn(5000, 6, generator=g)) - 6) < 0.2
    assert m.effective_rank(torch.randn(5000, 1, generator=g).repeat(1, 6)) < 1.05


def test_collapse_vs_vicreg():
    data = torch.randn(4000, 2, generator=torch.Generator().manual_seed(0))
    inv, _ = m.train_siamese(data, "invariance")
    vic, _ = m.train_siamese(data, "vicreg")
    with torch.no_grad():
        zi, zv = inv(data), vic(data)
    print(f"\nstd invariance {zi.std(0).mean():.4f}  vicreg {zv.std(0).mean():.3f}  erank vicreg {m.effective_rank(zv):.2f}")
    assert zi.std(0).mean() < 0.05, "invariance-only should collapse"
    assert zv.std(0).mean() > 0.5 and m.effective_rank(zv) > 1.5
