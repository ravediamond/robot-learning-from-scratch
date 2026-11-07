"""Run AFTER writing your guesses AND after your tests pass (~2 min)."""
import sys as _sys


def _unfinished_lab_hint(exc_type, exc, tb):
    if exc_type is NotImplementedError:
        print("\nThis predict script runs YOUR lab code, which is not finished yet (a TODO still raises NotImplementedError)."
              "\nMake `uv run pytest . -k \"not review\"` green in this folder first, then run me again.", file=_sys.stderr)
    else:
        _sys.__excepthook__(exc_type, exc, tb)


_sys.excepthook = _unfinished_lab_hint
import pixel_world as W
import w7d4_lewm as m

tr, va = W.make_episodes(200, 20, seed=0), W.make_episodes(40, 20, seed=1)
for lam, steps in [(0.0, 500), (0.1, 1500), (1.0, 1500)]:
    enc, pred = m.train_lewm(tr, lam, steps=steps)
    r = m.latent_report(enc, pred, tr, va)
    print(f"lambda {lam}: " + "  ".join(f"{k} {v:.4f}" for k, v in r.items()))
