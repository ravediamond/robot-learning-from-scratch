"""Run AFTER writing your guesses AND after your tests pass (~1 min)."""
import sys as _sys


def _unfinished_lab_hint(exc_type, exc, tb):
    if exc_type is NotImplementedError:
        print("\nThis predict script runs YOUR lab code, which is not finished yet (a TODO still raises NotImplementedError)."
              "\nMake `uv run pytest . -k \"not review\"` green in this folder first, then run me again.", file=_sys.stderr)
    else:
        _sys.__excepthook__(exc_type, exc, tb)


_sys.excepthook = _unfinished_lab_hint
import dagger_world as w
import w6d5_dagger as m

O, A = w.expert_demos(10, seed=0)
bc = m.train_bc(O, A)
r = m.run(bc, 200, seed=1)
print(f"1 BC 10 demos: success {r['success'].float().mean():.2f}  collision {r['collision'].float().mean():.2f}  "
      f"timeout {(~r['success'] & ~r['collision']).float().mean():.2f}")
for it in (1, 3):
    pol, n = m.dagger(O, A, iterations=it)
    print(f"2 DAgger {it} it ({n} labels): success {m.run(pol, 200, seed=1)['success'].float().mean():.2f}")
O60, A60 = w.expert_demos(60, seed=0)
print(f"3 BC 60 demos ({len(O60)} labels): success {m.run(m.train_bc(O60, A60), 200, seed=1)['success'].float().mean():.2f}")
