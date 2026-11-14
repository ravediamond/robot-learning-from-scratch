"""Run AFTER writing your guesses AND after your tests pass."""
import sys as _sys


def _unfinished_lab_hint(exc_type, exc, tb):
    if exc_type is NotImplementedError:
        print("\nThis predict script runs YOUR lab code, which is not finished yet (a TODO still raises NotImplementedError)."
              "\nMake `uv run pytest . -k \"not review\"` green in this folder first, then run me again.", file=_sys.stderr)
    else:
        _sys.__excepthook__(exc_type, exc, tb)


_sys.excepthook = _unfinished_lab_hint
import w12d3_failures as m

r1 = ["grasp"] * 50 + ["placement"] * 50
r2 = ["grasp"] * 40 + ["placement"] * 10 + ["grasp"] * 10 + ["placement"] * 40
print(f"1. kappa = {m.cohen_kappa(r1, r2):.2f}")
r1 = ["grasp"] * 90 + ["placement"] * 10
r2 = ["grasp"] * 80 + ["placement"] * 10 + ["grasp"] * 10
print(f"2. kappa = {m.cohen_kappa(r1, r2):.2f} (same 80% agreement, but chance agreement is now 82%)")
recs = [{"policy": "A", "outcome": "success", "failure_tag": None, "stage_reached": 4}] * 45 + \
       [{"policy": "A", "outcome": "failure", "failure_tag": "grasp", "stage_reached": 1}] * 5 + \
       [{"policy": "B", "outcome": "success", "failure_tag": None, "stage_reached": 4}] * 25 + \
       [{"policy": "B", "outcome": "failure", "failure_tag": "grasp", "stage_reached": 1}] * 10 + \
       [{"policy": "B", "outcome": "failure", "failure_tag": "placement", "stage_reached": 3}] * 15
for p in "AB":
    g = m.breakdown(recs, p)["grasp"]
    print(f"3. {p}: grasp failures {g['count']}/50 = {g['rate']:.0%} (CI {g['lo']:.0%}-{g['hi']:.0%})")
print("   share of failures: A 100% grasp, B 40% grasp: the share points at the wrong policy")
f = [{"policy": "C", "outcome": "failure", "failure_tag": "perception", "stage_reached": k} for k in [0] * 4 + [1] * 16 + [2] * 2 + [3] * 3] + \
    [{"policy": "C", "outcome": "success", "failure_tag": None, "stage_reached": 4}] * 25
print("4.", m.biggest_drop(m.stage_funnel(f, "C")))
