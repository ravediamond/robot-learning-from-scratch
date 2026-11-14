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

import fake_fleet
import w13d1_qa as m

eps, truth = fake_fleet.make_fleet(seed=0)
lengths = np.array([len(e["state"]) for e in eps], dtype=float)
z_naive = (lengths - lengths.mean()) / lengths.std()
print(f"1. mean +- 3 sd flags: {[eps[i]['id'] for i in np.where(np.abs(z_naive) > 3)[0]]} (the short one hides behind the long one)")
z = m.robust_z(lengths)
print(f"2. robust z flags: {[eps[i]['id'] for i in np.where(np.abs(z) > 3.5)[0]]}")
print("3. 300 / 30 = 10 s of a frozen image")
print(f"4. range of the dead joint: {np.ptp(eps[12]['state'][:, 2]):.3f}; inside the limits, so only a range check sees it")
rep = m.qa_report(eps, fake_fleet.LIMITS, fps=30)
print("5. duplicates:", rep["duplicates"], "(content hash: one pass, O(total frames))")
print("   quarantined:", rep["quarantine"])
