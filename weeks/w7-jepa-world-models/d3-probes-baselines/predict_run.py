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

import w7d3_probes as m

g = torch.Generator().manual_seed(0)
P_tr, P_va = torch.rand(2000, 2, generator=g) * 1.6 - 0.8, torch.rand(500, 2, generator=g) * 1.6 - 0.8
lat = lambda p: torch.stack([torch.sin(3.8 * p[:, 0]), torch.cos(3.8 * p[:, 0]), torch.sin(3.8 * p[:, 1]), torch.cos(3.8 * p[:, 1])], 1)
W, b = m.ridge_probe(lat(P_tr), P_tr)
print("1 linear R2", round(m.r2(m.apply_probe(lat(P_va), W, b), P_va), 3),
      " MLP R2", round(m.r2(m.mlp_probe(lat(P_tr), P_tr)(lat(P_va)).detach(), P_va), 3))
print("2 same-frame R2 is optimistic (it fits noise); held-out episodes are the honest number")
print("4 ratio 0.9: barely better than copy-last -> learned almost nothing about dynamics")
