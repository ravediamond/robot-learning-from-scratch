import warnings

import torch
from torch import nn

from w14d1_export import DeployablePolicy, export_onnx, parity, run_onnx

warnings.filterwarnings("ignore")


def make():
    torch.manual_seed(0)
    core = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 6))
    return DeployablePolicy(core, torch.arange(6.0), torch.full((6,), 2.0), torch.full((6,), -1.0), torch.full((6,), 0.1))


def test_wrapper_normalises_and_saves_stats():
    pol = make().eval()
    x = torch.randn(3, 6)
    with torch.no_grad():
        expected = pol.core((x - torch.arange(6.0)) / 2.0) * 0.1 - 1.0
        assert torch.allclose(pol(x), expected, atol=1e-6)
    sd = pol.state_dict()
    assert {"state_mean", "state_std", "action_mean", "action_std"} <= set(sd), "stats must be buffers"
    assert not any(n.endswith("_mean") or n.endswith("_std") for n, _ in pol.named_parameters()), "not trainable"


def test_export_parity_and_dynamic_batch(tmp_path):
    pol = make()
    path = tmp_path / "p.onnx"
    export_onnx(pol, torch.randn(1, 6), path)
    assert parity(pol, path, torch.randn(500, 6)) < 1e-5
    assert run_onnx(path, torch.randn(7, 6)).shape == (7, 6)
    assert not pol.training


def test_parity_uses_eval_mode(tmp_path):
    pol = make()
    path = tmp_path / "p.onnx"
    export_onnx(pol, torch.randn(1, 6), path)
    pol.train()
    assert parity(pol, path, torch.randn(64, 6)) < 1e-5


def test_fixed_batch_export(tmp_path):
    pol = make()
    path = tmp_path / "fixed.onnx"
    export_onnx(pol, torch.randn(1, 6), path, dynamic_batch=False)
    try:
        run_onnx(path, torch.randn(2, 6))
    except Exception:
        return
    raise AssertionError("a graph exported without dynamic axes should reject another batch size")
