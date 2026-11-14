"""Run AFTER writing your guesses AND after your tests pass."""
import sys as _sys


def _unfinished_lab_hint(exc_type, exc, tb):
    if exc_type is NotImplementedError:
        print("\nThis predict script runs YOUR lab code, which is not finished yet (a TODO still raises NotImplementedError)."
              "\nMake `uv run pytest . -k \"not review\"` green in this folder first, then run me again.", file=_sys.stderr)
    else:
        _sys.__excepthook__(exc_type, exc, tb)


_sys.excepthook = _unfinished_lab_hint
import torch

import w14d2_quant as m

torch.manual_seed(0)
w = 0.02 * torch.randn(64, 64)
w[0] *= 100
for per_channel in (False, True):
    q, s = m.quantize(w, per_channel=per_channel)
    err = (m.dequantize(q, s) - w)[1:].abs().mean() / w[1:].abs().mean()
    print(f"1. per_channel={per_channel}: relative error on the normal rows {err:.1%}")
a = torch.cat([torch.randn(10_000), torch.full((5,), 80.0)])
print(f"2. min-max scale {a.abs().max().item() / 127:.3f} vs percentile scale {m.percentile_scale(a, 99.9):.3f}")
x = torch.tensor([0.3, 1.7], requires_grad=True)
torch.round(x).sum().backward()
print(f"3. gradient of round: {x.grad.tolist()} (the STE replaces it by 1)")
print("4. 4x less memory. Speed only with int8 kernels (TensorRT, onnxruntime int8); otherwise dequantise and run fp32.")
