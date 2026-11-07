"""Run AFTER predict.md AND after your w2d5 tests pass (uses your code, ~1 min)."""
import sys as _sys


def _unfinished_lab_hint(exc_type, exc, tb):
    if exc_type is NotImplementedError:
        print("\nThis predict script runs YOUR lab code, which is not finished yet (a TODO still raises NotImplementedError)."
              "\nMake `uv run pytest . -k \"not review\"` green in this folder first, then run me again.", file=_sys.stderr)
    else:
        _sys.__excepthook__(exc_type, exc, tb)


_sys.excepthook = _unfinished_lab_hint
import torch

import w2d5_pixel_policy as m
from dot_world import make_dot_data

tr, va = make_dot_data(40, seed=0), make_dot_data(10, seed=1)
hold = m.masked_chunk_mse(va["state"][:, None, :].expand_as(va["actions"]), va["actions"], va["mask"]).item()
so = m.train_policy(tr, use_image=False, epochs=20)
px = m.train_policy(tr, use_image=True, epochs=20)
blank = dict(va, img=torch.zeros_like(va["img"]))
print(f"1/2 val MSE  hold {hold:.4f}  state-only {m.evaluate(so, va):.4f}  pixel {m.evaluate(px, va):.4f}")
print(f"3 pixel policy with blank images: {m.evaluate(px, blank):.4f}")
print("4 run train_real.py on your Mac")
