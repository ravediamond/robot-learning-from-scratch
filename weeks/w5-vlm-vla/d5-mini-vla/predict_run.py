"""Run AFTER writing your guesses AND after your tests pass (~2 min)."""
import sys as _sys


def _unfinished_lab_hint(exc_type, exc, tb):
    if exc_type is NotImplementedError:
        print("\nThis predict script runs YOUR lab code, which is not finished yet (a TODO still raises NotImplementedError)."
              "\nMake `uv run pytest . -k \"not review\"` green in this folder first, then run me again.", file=_sys.stderr)
    else:
        _sys.__excepthook__(exc_type, exc, tb)


_sys.excepthook = _unfinished_lab_hint
import torch

import vla_world as w
import w5d5_mini_vla as m

model = m.train_vla(w.make_vla_demos(400, seed=0))
act = lambda img, tok, st, g: model.act(img, tok, st, g)
show = lambda name, r: print(f"{name:28s} success {r['success'].float().mean():.2f}  other {r['other'].float().mean():.2f}")
show("1 right instruction", w.run_episodes(act, 200, seed=1))
show("1 swapped instruction", w.run_episodes(act, 200, seed=1, swap_instruction=True))
pad = lambda img, tok, st, g: model.act(img, torch.where(torch.arange(4) == 3, 5, tok), st, g)
show("2 '<pad>' colour (red=success)", w.run_episodes(pad, 200, seed=1))
blank = lambda img, tok, st, g: model.act(torch.zeros_like(img), tok, st, g)
show("3 blank image", w.run_episodes(blank, 200, seed=1))
two = lambda img, tok, st, g: model.act(img, tok, st, g, steps=2)
show("4 two flow steps", w.run_episodes(two, 200, seed=1))
