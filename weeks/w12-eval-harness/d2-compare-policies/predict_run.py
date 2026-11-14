"""Run AFTER writing your guesses AND after your tests pass."""
import sys as _sys


def _unfinished_lab_hint(exc_type, exc, tb):
    if exc_type is NotImplementedError:
        print("\nThis predict script runs YOUR lab code, which is not finished yet (a TODO still raises NotImplementedError)."
              "\nMake `uv run pytest . -k \"not review\"` green in this folder first, then run me again.", file=_sys.stderr)
    else:
        _sys.__excepthook__(exc_type, exc, tb)


_sys.excepthook = _unfinished_lab_hint
import random

import w12d2_compare as m

print(f"1. 1 - 0.95^6 = {1 - 0.95 ** 6:.2f}")
print("2.", m.holm({"a": 0.01, "b": 0.04, "c": 0.03}))
print(f"3. MDE at n=50, p=0.5: {m.min_detectable_difference(50, 0.5):.2f}   at n=200: {m.min_detectable_difference(200, 0.5):.2f}")
print(f"4. MDE at n=50, p=0.8 (only upward is possible near the ceiling): {m.min_detectable_difference(50, 0.8):.2f}")
rng = random.Random(0)
same = [1 if rng.random() < 0.6 else 0 for _ in range(50)]
twin = [x if rng.random() < 0.8 else 1 - x for x in same]
print("5.", {k: round(v, 2) for k, v in m.rank_probabilities({"A": same, "B": twin}, 2000, 0).items()})
