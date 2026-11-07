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

import w7d2_sigreg as m

g = torch.Generator().manual_seed(0)
z = torch.randn(512, 16, generator=g)
Q, _ = torch.linalg.qr(torch.randn(16, 16, generator=g))
for name, x in [("N(0,I)", z), ("constant", torch.ones(512, 16)), ("std 3", 3 * z), ("rotated", z @ Q), ("mean +1", z + 1)]:
    print(f"1/2 {name:9s} SIGReg {m.sigreg(x, 64, torch.Generator().manual_seed(1)).item():.3f}")
print("3 (1 - m)^k = e^-1 -> k ~ 1/(1-m) = 100 steps")
