"""REVIEW. Averages a monitor's scores over 5 seeds per episode before computing AUROC. The result
does not match the paper's 0.8377 for W1 and changes when the CSV rows are reordered.
Find the bug, fix, replace:
BUG:
"""
import csv

import torch


def seed_averaged_scores(path, feature: str) -> tuple[torch.Tensor, torch.Tensor]:
    rows = [r for r in csv.DictReader(open(path)) if r["feature"] == feature]
    n_seeds = len({r["seed"] for r in rows})
    scores = torch.tensor([float(r["score"]) for r in rows]).view(-1, n_seeds).mean(1)
    labels = torch.tensor([int(r["label"]) for r in rows]).view(-1, n_seeds)[:, 0]
    return scores, labels
