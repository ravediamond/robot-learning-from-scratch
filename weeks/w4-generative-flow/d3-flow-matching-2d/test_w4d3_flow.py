import pytest
import torch

import w4d3_flow as m
from toy2d import mean_nn_distance, two_moons


def test_interpolate():
    x0, x1 = torch.zeros(3, 2), torch.ones(3, 2)
    xt, v = m.fm_interpolate(x0, x1, torch.tensor([0.0, 0.25, 1.0]))
    assert torch.allclose(xt[:, 0], torch.tensor([0.0, 0.25, 1.0])) and torch.allclose(v, torch.ones(3, 2))


def test_euler_on_known_field():
    class ToData(torch.nn.Module):            # exact velocity field for a point mass at c=(2,-1) from x0
        def forward(self, x, t):
            c = torch.tensor([2.0, -1.0])
            return (c - x) / (1 - t[:, None]).clamp(min=1e-6)
    x0 = torch.randn(5, 2)
    out = m.euler_sample(ToData(), 5, 4, torch.Generator(), x0=x0)
    assert torch.allclose(out, torch.tensor([[2.0, -1.0]]).expand(5, 2), atol=1e-4)


@pytest.fixture(scope="module")
def trained():
    data = two_moons(4000, torch.Generator().manual_seed(0))
    torch.manual_seed(0)
    return data, m.train_fm(data)


def test_ten_steps_on_data(trained):
    data, model = trained
    s = m.euler_sample(model, 1000, 10, torch.Generator().manual_seed(1))
    d = mean_nn_distance(s, data)
    print(f"\n10-step nn-distance {d:.3f}")
    assert s.shape == (1000, 2) and d < 0.07


def test_one_step_collapses_to_mean(trained):
    data, model = trained
    s1 = m.euler_sample(model, 1000, 1, torch.Generator().manual_seed(1))
    print(f"1-step sample std {s1.std(0).tolist()} vs data std {data.std(0).tolist()}")
    assert (s1.std(0) < 0.35 * data.std(0)).all(), "1 step should land near the data mean"
    assert (s1.mean(0) - data.mean(0)).norm() < 0.15
