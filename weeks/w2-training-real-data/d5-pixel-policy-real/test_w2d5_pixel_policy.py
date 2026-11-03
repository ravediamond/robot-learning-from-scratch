import pytest
import torch

import w2d5_pixel_policy as m
from dot_world import make_dot_data


def test_masked_chunk_mse():
    p, t = torch.ones(2, 3, 2), torch.zeros(2, 3, 2)
    mask = torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.bool)
    t[1, 1:] = 99.0
    assert torch.isclose(m.masked_chunk_mse(p, t, mask), torch.tensor(1.0))


def test_policy_shapes_and_buffers():
    stats = {"state_mean": torch.zeros(2), "state_std": torch.ones(2),
             "act_mean": torch.zeros(2), "act_std": torch.ones(2)}
    pol = m.PixelStatePolicy(stats, H=4)
    assert pol(torch.rand(5, 3, 32, 32), torch.randn(5, 2)).shape == (5, 4, 2)
    assert {"state_mean", "act_std"} <= set(pol.state_dict())
    so = m.PixelStatePolicy(stats, H=4, use_image=False)
    assert not any(isinstance(x, torch.nn.Conv2d) for x in so.modules()), "state-only must have no CNN"
    out1 = so(torch.rand(2, 3, 32, 32), torch.zeros(2, 2))
    out2 = so(torch.zeros(2, 3, 32, 32), torch.zeros(2, 2))
    assert torch.equal(out1, out2)


@pytest.fixture(scope="module")
def data():
    return make_dot_data(40, seed=0), make_dot_data(10, seed=1)


def test_pixels_beat_state_only(data):
    tr, va = data
    torch.manual_seed(0)
    so = m.train_policy(tr, use_image=False, epochs=20)
    px = m.train_policy(tr, use_image=True, epochs=20)
    e_so, e_px = m.evaluate(so, va), m.evaluate(px, va)
    print(f"\nval MSE state-only {e_so:.4f}  pixel {e_px:.4f}")
    assert not px.training
    assert e_px < 0.3 * e_so, "the image branch is not being used"
