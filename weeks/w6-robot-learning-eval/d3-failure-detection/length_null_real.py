"""PROVIDED driver (needs internet): the length null on YOUR labelled rollouts, with YOUR functions.
    uv run python length_null_real.py
Compare with the anchor monitors in reference/journal/papers/jepa-world-model/gaps/anchor_ci.json
(W1 seed-averaged AUROC 0.84 [0.73, 0.93] on 65 scored episodes)."""
import csv

import torch
from huggingface_hub import hf_hub_download

import w6d3_monitors as m

path = hf_hub_download("ravediamond/so101-bowls-policy-rollouts", "meta/rollout_labels.csv", repo_type="dataset")
rows = [r for r in csv.DictReader(open(path)) if r.get("outcome") in ("success", "failure")]
length = torch.tensor([float(r["length"]) for r in rows])
label = torch.tensor([1 if r["outcome"] == "failure" else 0 for r in rows])
print(f"{len(rows)} labelled episodes, {int(label.sum())} failures")
a = m.auroc(length, label)
lo, hi = m.auroc_ci(length, label, 2000, seed=0)
print(f"length-null AUROC {a:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
print(f"TPR at 10% FPR: {m.tpr_at_fpr(length, label, 0.10):.2f}")
