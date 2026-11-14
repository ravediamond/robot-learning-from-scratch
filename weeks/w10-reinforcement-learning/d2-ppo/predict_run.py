"""Run AFTER writing your guesses AND after your tests pass (~30 s)."""
import sys as _sys


def _unfinished_lab_hint(exc_type, exc, tb):
    if exc_type is NotImplementedError:
        print("\nThis predict script runs YOUR lab code, which is not finished yet (a TODO still raises NotImplementedError)."
              "\nMake `uv run pytest . -k \"not review\"` green in this folder first, then run me again.", file=_sys.stderr)
    else:
        _sys.__excepthook__(exc_type, exc, tb)


_sys.excepthook = _unfinished_lab_hint
import torch

import rl_envs
import w10d2_ppo as m

for A in (1.0, -1.0):
    for rho in (0.5, 1.0, 1.5):
        r = torch.tensor([rho], requires_grad=True)
        obj = -m.ppo_clip_loss(r.log(), torch.zeros(1), torch.tensor([A]), 0.2)
        obj.backward()
        print(f"A={A:+.0f} rho={rho}: objective {obj.item():+.2f}  d/drho {r.grad.item():+.1f}")
rand = rl_envs.evaluate(lambda o: torch.rand(len(o), 2) * 2 - 1)["return"].mean().item()
exp = rl_envs.evaluate(rl_envs.expert_action)["return"].mean().item()
policy, hist = m.train_ppo(iters=40, seed=0)
got = rl_envs.evaluate(lambda o: policy.dist(o).mean)["return"].mean().item()
print(f"4. random {rand:.1f}   expert {exp:.1f}   PPO (mean action) {got:.1f}")
print("5. the value network gets a gradient from the POLICY loss and learns to make advantages look good")
