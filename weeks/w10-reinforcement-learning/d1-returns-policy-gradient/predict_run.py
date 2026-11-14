"""Run AFTER writing your guesses AND after your tests pass (~20 s)."""
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
import w10d1_pg as m

r = torch.ones(3, 1)
print("1.", m.discounted_returns(r, torch.zeros(3, 1, dtype=torch.bool), 0.9).flatten().tolist())
print("2.", m.discounted_returns(r, torch.tensor([[False], [True], [False]]), 0.9).flatten().tolist())
print("3. lambda=1: G - V (Monte-Carlo advantage); lambda=0: the one-step TD error delta")
print("4. every log-prob is pushed DOWN, by more when the return was worse: correct in expectation, very noisy")


def grad_var(use_baseline):
    grads = []
    for seed in range(30):
        torch.manual_seed(0)
        pol = m.GaussianPolicy(4, 2)
        env = rl_envs.Reach2D(64, seed=seed)
        logp, rew, done = m.collect(env, pol)
        G = m.discounted_returns(rew, done, 0.99)
        b = G.mean(dim=1, keepdim=True) if use_baseline else None
        pol.zero_grad()
        m.reinforce_loss(logp, G, b).backward()
        grads.append(torch.cat([p.grad.flatten() for p in pol.parameters()]))
    return torch.stack(grads).var(dim=0).sum().item()


v0, v1 = grad_var(False), grad_var(True)
print(f"5. gradient variance without baseline {v0:.1f}, with {v1:.1f}: ratio {v0 / v1:.0f}x")
for ub in (False, True):
    hist = m.train_reinforce(iters=150, use_baseline=ub, seed=0)
    print(f"   baseline={ub}: mean return {hist[0]:.1f} -> {hist[-1]:.1f}")
