"""Run AFTER writing your guesses AND after your tests pass (~40 s)."""
import sys as _sys


def _unfinished_lab_hint(exc_type, exc, tb):
    if exc_type is NotImplementedError:
        print("\nThis predict script runs YOUR lab code, which is not finished yet (a TODO still raises NotImplementedError)."
              "\nMake `uv run pytest . -k \"not review\"` green in this folder first, then run me again.", file=_sys.stderr)
    else:
        _sys.__excepthook__(exc_type, exc, tb)


_sys.excepthook = _unfinished_lab_hint
import math

import torch

import rl_envs
import w10d3_sac as m

print(f"1. log N(0;0,1) = {-0.5 * math.log(2 * math.pi):.3f}, correction = -log(1 - 0) = 0, log pi = {-0.5 * math.log(2 * math.pi):.3f}")
print(f"2. u=3: -log(1 - tanh(3)^2) = {-math.log(1 - math.tanh(3) ** 2):+.2f}: the density piles up near the bound")
print(f"3. about 1/tau = 200 updates ((1 - 0.005)^200 = {0.995 ** 200:.2f} left)")
print("4. Q-learning maximises over noisy estimates, which biases them upward; the minimum is a cheap pessimistic fix")
actor, hist = m.train_sac(steps=1500, seed=0)
ret = rl_envs.evaluate(lambda o: torch.tanh(actor(o)[0]))["return"].mean().item()
print(f"5. SAC deterministic return {ret:.1f}")
