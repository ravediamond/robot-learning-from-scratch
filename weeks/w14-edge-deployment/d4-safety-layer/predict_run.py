"""Run AFTER writing your guesses AND after your tests pass."""
import sys as _sys


def _unfinished_lab_hint(exc_type, exc, tb):
    if exc_type is NotImplementedError:
        print("\nThis predict script runs YOUR lab code, which is not finished yet (a TODO still raises NotImplementedError)."
              "\nMake `uv run pytest . -k \"not review\"` green in this folder first, then run me again.", file=_sys.stderr)
    else:
        _sys.__excepthook__(exc_type, exc, tb)


_sys.excepthook = _unfinished_lab_hint
import numpy as np

import w14d4_safety as m

print("1.", m.rate_limit(np.array([0.5]), np.array([0.9]), 0.05))
print(f"2. capture-time stale: {m.is_stale(10.00, now=10.76, max_age=0.2)}; arrival-time would say: {m.is_stale(10.75, now=10.76, max_age=0.2)}")
f = m.SafetyFilter(np.array([[-1.0, 1.0]]), max_step=0.1, max_age=0.2, margin=0.0)
print("3.", f.step(np.array([np.nan]), capture_time=0.0, now=0.01), "(hold the last safe command)")
f.estop()
print("4.", f.step(np.array([0.05]), capture_time=1.0, now=1.01), "(latched until reset())")
