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

import w13d5_flywheel as m

flags = np.array([int(c) for c in "0001111000011"])
print("1.", m.intervention_segments(flags, context=2))
print("2. the executed (human) action: the policy's proposal is what went wrong.")
print("3.", m.allocate_collection([0.1, 0.5, 0.2, 0.0, 0.6, 0.3, 0.1, 0.05], [10] * 8, 40))
