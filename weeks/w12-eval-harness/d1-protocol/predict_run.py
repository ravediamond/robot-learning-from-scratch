"""Run AFTER writing your guesses AND after your tests pass."""
import sys as _sys


def _unfinished_lab_hint(exc_type, exc, tb):
    if exc_type is NotImplementedError:
        print("\nThis predict script runs YOUR lab code, which is not finished yet (a TODO still raises NotImplementedError)."
              "\nMake `uv run pytest . -k \"not review\"` green in this folder first, then run me again.", file=_sys.stderr)
    else:
        _sys.__excepthook__(exc_type, exc, tb)


_sys.excepthook = _unfinished_lab_hint
import w12d1_protocol as m

sched = m.make_schedule(["A", "B", "C", "D"], n_starts=10, repeats=5, seed=0)
print(f"1. {len(sched)} rollouts in {len({r['block'] for r in sched})} blocks")
N = 2 * 50
drift = lambda i: 0.9 - 0.4 * i / (N - 1)
seq = [{"trial": i, "policy": "A" if i < 50 else "B"} for i in range(N)]
print("2. one after the other:", {k: round(v, 3) for k, v in m.expected_success(seq, drift).items()})
blocked = m.make_schedule(["A", "B"], n_starts=10, repeats=5, seed=0)
print("3. blocked:", {k: round(v, 3) for k, v in m.expected_success(blocked, drift).items()})
first = {p: 0 for p in "ABCD"}
blocks = {}
for r in sched:
    blocks.setdefault(r["block"], []).append(r)
for rows in blocks.values():
    first[min(rows, key=lambda r: r["trial"])["policy"]] += 1
print("4. times first in block:", first, "(fixed order: one policy is first 50 times)")
print("5. reset the scene more carefully, wait a little longer before calling a failure, nudge the object, "
      "intervene earlier or later, retry a 'glitch'.")
