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

import w13d2_curation as m

groups = [f"s{i // 10}" for i in range(200)]
rng = np.random.default_rng(0)
val = set(rng.choice(200, 40, replace=False).tolist())
train_sessions = {groups[i] for i in range(200) if i not in val}
print(f"1. random split: {sum(1 for i in val if groups[i] in train_sessions)} of 40 validation episodes share a session with train")
tr, va = m.group_split(groups, 0.2, seed=0)
print(f"2. group split: {len({groups[i] for i in tr} & {groups[i] for i in va})} shared sessions")
t = np.linspace(0, 1, 240)[:, None] * np.ones((1, 6))
u = np.linspace(0, 1, 300)[:, None] * np.ones((1, 6))
print(f"3. distance after resampling: {np.linalg.norm(m.resample(t, 50) - m.resample(u, 50)):.2e}")
print("4. add+remove: changes; reorder: no change (sorted hashes)")
