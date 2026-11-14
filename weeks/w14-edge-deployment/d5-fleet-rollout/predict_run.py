"""Run AFTER writing your guesses AND after your tests pass."""
import sys as _sys


def _unfinished_lab_hint(exc_type, exc, tb):
    if exc_type is NotImplementedError:
        print("\nThis predict script runs YOUR lab code, which is not finished yet (a TODO still raises NotImplementedError)."
              "\nMake `uv run pytest . -k \"not review\"` green in this folder first, then run me again.", file=_sys.stderr)
    else:
        _sys.__excepthook__(exc_type, exc, tb)


_sys.excepthook = _unfinished_lab_hint
import w14d5_rollout as m

for k in (3, 12):
    print(f"canary {k}/40h vs baseline 30/400h: p = {m.rate_ratio_pvalue(k, 40, 30, 400):.4f} -> "
          f"{m.canary_decision(k, 40, 30, 400)}")
print("3. not necessarily: it shows WHERE they differ. Look at those steps (videos), then let the canary decide.")
print("4.", m.stage_sizes(50, [0.05, 0.25, 1.0]))
