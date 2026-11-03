import copy

import torch
import torch.nn as nn

import w7d2_sigreg as m


def test_epps_pulley_shapes_and_order():
    g = torch.Generator().manual_seed(0)
    gauss = torch.randn(1000, 3, generator=g)
    ep = m.epps_pulley(gauss)
    assert ep.shape == (3,) and (ep < 1.5).all(), "N * statistic is O(1) for true Gaussians"
    assert (m.epps_pulley(torch.zeros(1000, 2)) > 50).all()
    assert (m.epps_pulley(3 * gauss) > 5 * m.epps_pulley(gauss)).all()


def test_sigreg_detects_collapse_and_scale():
    g = torch.Generator().manual_seed(0)
    z = torch.randn(512, 16, generator=g)
    ok = m.sigreg(z, 64, torch.Generator().manual_seed(1)).item()
    assert ok < 2.0
    assert m.sigreg(torch.ones(512, 16), 64, torch.Generator().manual_seed(1)).item() > 50
    assert m.sigreg(z + 1.0, 64, torch.Generator().manual_seed(1)).item() > 5 * ok
    zz = torch.randn(256, 8, requires_grad=True)
    m.sigreg(zz, 16, torch.Generator().manual_seed(2)).backward()
    assert zz.grad is not None and torch.isfinite(zz.grad).all(), "must be differentiable"


def test_ema():
    online, target = nn.Linear(3, 3), nn.Linear(3, 3)
    with torch.no_grad():
        online.weight.fill_(1.0)
        target.weight.fill_(0.0)
    m.ema_update(target, online, 0.9)
    assert torch.allclose(target.weight, torch.full((3, 3), 0.1))
    assert torch.allclose(online.weight, torch.ones(3, 3)), "online must not change"


def test_stop_gradient():
    torch.manual_seed(0)
    online, pred = nn.Linear(4, 4), nn.Linear(4, 4)
    target = copy.deepcopy(online)
    loss = m.jepa_ema_loss(online, target, pred, torch.randn(8, 4), torch.randn(8, 4))
    loss.backward()
    assert online.weight.grad is not None and pred.weight.grad is not None
    assert target.weight.grad is None, "no gradient may reach the target encoder"
