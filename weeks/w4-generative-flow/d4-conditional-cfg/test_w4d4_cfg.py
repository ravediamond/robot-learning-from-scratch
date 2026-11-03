import pytest
import torch

import w4d4_cfg as m
from toy2d import CENTRES, clusters


def test_drop_labels():
    c = torch.zeros(10000, dtype=torch.long)
    d = m.drop_labels(c, 0.1, 4, torch.Generator().manual_seed(0))
    assert abs((d == 4).float().mean().item() - 0.1) < 0.02 and (c == 0).all()


def test_cfg_formula():
    class Fake(torch.nn.Module):
        null = 4

        def forward(self, x, t, c):
            return torch.where((c == self.null)[:, None], torch.zeros_like(x), torch.ones_like(x))
    x, t, c = torch.zeros(3, 2), torch.zeros(3), torch.tensor([0, 1, 2])
    assert torch.allclose(m.cfg_velocity(Fake(), x, t, c, 0.0), torch.zeros(3, 2))
    assert torch.allclose(m.cfg_velocity(Fake(), x, t, c, 1.0), torch.ones(3, 2))
    assert torch.allclose(m.cfg_velocity(Fake(), x, t, c, 3.0), torch.full((3, 2), 3.0))


@pytest.fixture(scope="module")
def model():
    x, y = clusters(4000, torch.Generator().manual_seed(0))
    torch.manual_seed(0)
    return m.train_cond_fm(x, y)


def _hit(s, req):
    return (torch.cdist(s, CENTRES).argmin(1) == req).float().mean().item()


def test_conditional_and_unconditional(model):
    req = torch.arange(4).repeat(250)
    s1 = m.cfg_sample(model, req, 10, 1.0, torch.Generator().manual_seed(1))
    s0 = m.cfg_sample(model, req, 10, 0.0, torch.Generator().manual_seed(1))
    print(f"\nhit rate w=1 {_hit(s1, req):.2f}  w=0 {_hit(s0, req):.2f}")
    assert _hit(s1, req) > 0.95
    assert _hit(s0, req) < 0.45, "w=0 must ignore the condition"
    frac = torch.bincount(torch.cdist(s0, CENTRES).argmin(1), minlength=4).float() / len(s0)
    assert (frac > 0.12).all(), "unconditional samples should cover all four corners"
