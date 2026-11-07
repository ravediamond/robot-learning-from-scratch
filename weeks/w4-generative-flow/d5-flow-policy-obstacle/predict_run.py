"""Run AFTER writing your guesses AND after your tests pass (~2 min)."""
import sys as _sys


def _unfinished_lab_hint(exc_type, exc, tb):
    if exc_type is NotImplementedError:
        print("\nThis predict script runs YOUR lab code, which is not finished yet (a TODO still raises NotImplementedError)."
              "\nMake `uv run pytest . -k \"not review\"` green in this folder first, then run me again.", file=_sys.stderr)
    else:
        _sys.__excepthook__(exc_type, exc, tb)


_sys.excepthook = _unfinished_lab_hint
import torch

import sim2d
import w4d5_flow_policy as m

d = sim2d.make_demos(200, 8, seed=0)
mse, flow = m.train_mse_policy(d, 8), m.train_flow_policy(d, 8)
one_step = lambda obs, g: flow.act(obs, g, steps=1)
for name, fn, hx in [("MSE", mse.act, 4), ("flow 10 steps", flow.act, 4), ("flow 1 step", one_step, 4),
                     ("flow replan every step", flow.act, 1)]:
    r = m.rollout(fn, 200, hx, seed=1)
    k = int(r["collision"].sum())
    print(f"{name:24s} collision {k / 200:.2f} {tuple(round(x, 2) for x in sim2d.wilson(k, 200))}  success {r['success'].float().mean():.2f}")
