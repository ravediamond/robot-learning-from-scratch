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

import reward_data
import w11d1_rewards as m

tr, va = reward_data.classifier_data(seed=0), reward_data.classifier_data(seed=1)
clf = m.train_success_classifier(tr["x"], tr["y"], seed=0)
with torch.no_grad():
    p = torch.sigmoid(clf(va["x"]).squeeze(-1))
    print(f"1. held-out accuracy {((p > 0.5) == va['y'].bool()).float().mean():.3f}")
    far = va["x"][va["dist"] > 0.3].clone()
    far[:, 4] = 1.0
    print(f"2. far from the goal, flag set: called success {(torch.sigmoid(clf(far).squeeze(-1)) > 0.5).float().mean():.2f}")
    thr = m.threshold_for_precision(p, va["y"], 0.99)
    rec = ((p >= thr) & va["y"].bool()).sum() / va["y"].sum()
    print(f"3. threshold {thr:.2f} for precision 0.99: recall {rec:.2f}")
ep, te = reward_data.expert_episodes(seed=0), reward_data.expert_episodes(seed=1)
model = m.train_progress_model(ep["x"], ep["episode"], ep["t"], seed=0)
with torch.no_grad():
    print(f"4. Spearman(score, -distance) on new episodes: {m.spearman(model(te['x']).squeeze(-1), -te['dist']):.2f}")
bad = m.train_progress_model(ep["x"], torch.zeros_like(ep["episode"]), ep["t"], seed=0)
with torch.no_grad():
    print(f"5. pairs across episodes: {m.spearman(bad(te['x']).squeeze(-1), -te['dist']):.2f}"
          "  (learns 'time since start', which is only partly progress)")
