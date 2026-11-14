import torch
from torch import nn

from w14d2_quant import dequantize, fake_quant, percentile_scale, quantize, quantize_linear_weights


def test_quantize_round_trip():
    torch.manual_seed(0)
    w = torch.randn(8, 16)
    q, s = quantize(w)
    assert q.dtype == torch.int8 and s.shape == (8, 1) and q.abs().max() <= 127
    assert (dequantize(q, s) - w).abs().max() <= s.max() / 2 + 1e-7
    q1, s1 = quantize(w, per_channel=False)
    assert s1.dim() == 0
    z = torch.zeros(2, 3)
    z[1] = 1.0
    qz, sz = quantize(z)
    assert sz[0].item() == 1.0 and (qz[0] == 0).all() and (qz[1] == 127).all()
    q4, s4 = quantize(torch.randn(4, 3, 3, 3))
    assert s4.shape == (4, 1, 1, 1)


def test_per_channel_beats_per_tensor_with_an_outlier_row():
    torch.manual_seed(0)
    w = 0.02 * torch.randn(64, 64)
    w[0] *= 100
    errs = {}
    for pc in (False, True):
        q, s = quantize(w, per_channel=pc)
        errs[pc] = ((dequantize(q, s) - w)[1:].abs().mean() / w[1:].abs().mean()).item()
    assert errs[True] < 0.01 and errs[False] > 0.2


def test_percentile_clips_outliers():
    torch.manual_seed(0)
    a = torch.cat([torch.randn(10_000), torch.full((5,), 80.0)])
    assert percentile_scale(a, 99.9) < 0.04 < a.abs().max().item() / 127


def test_fake_quant_forward_and_gradient():
    x = torch.tensor([0.004, 0.026, 1.0, -0.05], requires_grad=True)
    y = fake_quant(x, 0.01)
    assert torch.allclose(y, torch.tensor([0.0, 0.03, 1.0, -0.05]))
    y.sum().backward()
    assert torch.equal(x.grad, torch.ones(4)), "straight-through: identity gradient"
    big = fake_quant(torch.tensor([5.0]), 0.01)
    assert abs(big.item() - 1.27) < 1e-6, "clamped to 127 levels"


def test_quantize_model_small_action_error():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(12, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 6))
    x = torch.randn(256, 12)
    with torch.no_grad():
        ref = model(x)
    errs = quantize_linear_weights(model)
    assert set(errs) == {"0", "2", "4"} and max(errs.values()) < 0.01
    with torch.no_grad():
        assert ((model(x) - ref).abs().max() / ref.abs().max()).item() < 0.02
