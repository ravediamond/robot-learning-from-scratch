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

import w13d3_mixtures as m

sizes = {"public": 1000, "ours": 100, "sim": 10}
for T in (1, 3):
    print(f"1. T={T}:", {k: round(v, 3) for k, v in m.temperature_weights(sizes, T).items()})
print("2.", {k: round(v, 1) for k, v in m.epochs_per_source(sizes, m.temperature_weights(sizes, 3), 200).items()})
a, b = m.fit_power_law(np.array([100.0, 400.0]), np.array([0.40, 0.20]))
print(f"3. b = {b:.2f}; error at 1600 = {a * 1600 ** -b:.2f}")
print("4.", m.ablation_effect([0.62, 0.70, 0.55], [0.58, 0.66, 0.50]))
