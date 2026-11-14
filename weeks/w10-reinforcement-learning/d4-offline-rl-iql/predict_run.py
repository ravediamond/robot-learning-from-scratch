"""Run AFTER writing your guesses AND after your tests pass (~40 s)."""
import sys as _sys


def _unfinished_lab_hint(exc_type, exc, tb):
    if exc_type is NotImplementedError:
        print("\nThis predict script runs YOUR lab code, which is not finished yet (a TODO still raises NotImplementedError)."
              "\nMake `uv run pytest . -k \"not review\"` green in this folder first, then run me again.", file=_sys.stderr)
    else:
        _sys.__excepthook__(exc_type, exc, tb)


_sys.excepthook = _unfinished_lab_hint
import torch

import offline_data
import rl_envs
import w10d4_iql as m

print("1. tau=0.8: weight 0.8 on d=+1 (value too low), 0.2 on d=-1")
x = torch.tensor([0.0, 0.0, 0.0, 10.0])
for tau in (0.5, 0.9):
    c = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([c], lr=0.05)
    for _ in range(2000):
        opt.zero_grad()
        m.expectile_loss(x - c, tau).backward()
        opt.step()
    print(f"2. tau={tau}: best constant {c.item():.2f}")
data = offline_data.make_dataset(seed=0)
exp = rl_envs.expert_action(data["obs"])
scale = (data["act"] * exp).sum() / (exp * exp).sum()
print(f"3. BC mean action = {scale.item():+.2f} x expert")
bc = m.train_bc(data, seed=0)
iql = m.train_iql(data, seed=0)
still = rl_envs.evaluate(lambda o: torch.zeros(len(o), 2))["return"].mean().item()
print(f"4. standing still {still:.1f}   BC {rl_envs.evaluate(bc)['return'].mean():.1f}   IQL {rl_envs.evaluate(iql)['return'].mean():.1f}")
print("5. beta -> 0: all weights 1: plain behavior cloning")
