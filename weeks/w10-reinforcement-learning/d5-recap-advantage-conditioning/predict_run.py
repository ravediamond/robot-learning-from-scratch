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
import w10d5_recap as m

data = offline_data.make_dataset(seed=0, per_step=True)
V = m.fit_value(data, seed=0)
adv = m.n_step_advantage(data, V, n=1, gamma=0.95)
ind = m.improvement_indicator(adv, torch.zeros(len(adv), dtype=torch.long), q=0.7)
names = ["expert", "random", "anti-expert"]
for k in range(3):
    print(f"2. {names[k]:12s} steps with I=1: {ind[data['kind'] == k].float().mean():.2f}")
pol = m.train_conditioned(data, ind, seed=0)
for i in (1, 0):
    print(f"3. condition I={i}: return {rl_envs.evaluate(lambda o: pol(o, i))['return'].mean():.1f}")
bc = m.train_conditioned(data, torch.ones_like(ind), seed=0)
print(f"   plain BC: return {rl_envs.evaluate(lambda o: bc(o, 1))['return'].mean():.1f}")
print("4. the model learns what 'worse than usual' looks like, and all data still trains the shared features")
adv5 = m.n_step_advantage(data, V, n=5, gamma=0.95)
ind5 = m.improvement_indicator(adv5, torch.zeros(len(adv5), dtype=torch.long), q=0.7)
print(f"5. expert steps with I=1: n=1 {ind[data['kind'] == 0].float().mean():.2f}, n=5 {ind5[data['kind'] == 0].float().mean():.2f}"
      " (the 4 following steps come from OTHER actors here: their luck is credited to this action)")
print("6. every step of the easy task is 'good' and every step of the hard one 'bad': the bit encodes the task, not the quality")
