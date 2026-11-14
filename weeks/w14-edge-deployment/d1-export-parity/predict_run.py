"""Run AFTER writing your guesses AND after your tests pass."""
import sys as _sys


def _unfinished_lab_hint(exc_type, exc, tb):
    if exc_type is NotImplementedError:
        print("\nThis predict script runs YOUR lab code, which is not finished yet (a TODO still raises NotImplementedError)."
              "\nMake `uv run pytest . -k \"not review\"` green in this folder first, then run me again.", file=_sys.stderr)
    else:
        _sys.__excepthook__(exc_type, exc, tb)


_sys.excepthook = _unfinished_lab_hint
import tempfile
import warnings
from pathlib import Path

import torch
from torch import nn

import w14d1_export as m

warnings.filterwarnings("ignore")
torch.manual_seed(0)
core = nn.Sequential(nn.Linear(6, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 6))
pol = m.DeployablePolicy(core, torch.zeros(6), torch.full((6,), 0.5), torch.zeros(6), torch.full((6,), 0.2))
print("1. inputs 2x too large and outputs 5x too small, in the wrong frame: it moves somewhere, confidently")
with tempfile.TemporaryDirectory() as d:
    p = Path(d) / "policy.onnx"
    m.export_onnx(pol, torch.randn(1, 6), p, dynamic_batch=False)
    try:
        m.run_onnx(p, torch.randn(2, 6))
    except Exception as e:
        print(f"2. fixed batch: {type(e).__name__}")
    m.export_onnx(pol, torch.randn(1, 6), p)
    print(f"3. parity: {m.parity(pol, p, torch.randn(256, 6)):.1e}")
    pol.train()
    with torch.no_grad():
        err = (pol(torch.ones(4, 6)) - torch.from_numpy(m.run_onnx(p, torch.ones(4, 6)))).abs().max().item()
    print(f"4. torch in train mode vs the graph: {err:.2f} (dropout active in torch only): parity() must call eval()")
