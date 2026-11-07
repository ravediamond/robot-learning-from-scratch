"""Run AFTER writing your guesses AND after your tests pass."""
import sys as _sys


def _unfinished_lab_hint(exc_type, exc, tb):
    if exc_type is NotImplementedError:
        print("\nThis predict script runs YOUR lab code, which is not finished yet (a TODO still raises NotImplementedError)."
              "\nMake `uv run pytest . -k \"not review\"` green in this folder first, then run me again.", file=_sys.stderr)
    else:
        _sys.__excepthook__(exc_type, exc, tb)


_sys.excepthook = _unfinished_lab_hint
import math

import torch

import w6d2_stats as m

for k, n in [(9, 10), (18, 20), (90, 100)]:
    print(f"1 Wilson {k}/{n}: {tuple(round(x, 3) for x in m.wilson(k, n))}")
print("2 normal approx 0/10: (0, 0) <- absurd; Wilson:", tuple(round(x, 3) for x in m.wilson(0, 10)))
print("3 McNemar(5, 1):", round(m.mcnemar_exact(5, 1), 4))
print("4", m.trials_needed(0.6, 0.8), m.trials_needed(0.85, 0.95))
g = torch.Generator().manual_seed(0)
ep_succ = (torch.rand(50, generator=g) < 0.7).float()
frames = ep_succ.repeat_interleave(300)
ep_ci = m.bootstrap_ci(ep_succ, lambda x: x.mean(), 2000, seed=0)
fr_ci = m.bootstrap_ci(frames, lambda x: x.mean(), 2000, seed=0)
print(f"5 episode CI {tuple(round(x, 3) for x in ep_ci)}  frame CI {tuple(round(x, 3) for x in fr_ci)}"
      f"  width ratio {(ep_ci[1] - ep_ci[0]) / (fr_ci[1] - fr_ci[0]):.1f} (~sqrt(300) = {math.sqrt(300):.1f})")
