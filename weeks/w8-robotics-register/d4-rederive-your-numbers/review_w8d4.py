"""REVIEW. SOLUTION.

BUG: view(-1, n_seeds) assumes rows are ordered episode-major (all seeds of episode 0, then episode 1...).
The CSV is seed-major, so each "row" of the view mixes different episodes. Group explicitly by episode.
"""
import csv
from collections import defaultdict

import torch


def seed_averaged_scores(path, feature):
    by_ep, lab = defaultdict(list), {}
    for r in csv.DictReader(open(path)):
        if r["feature"] == feature:
            by_ep[int(r["episode"])].append(float(r["score"]))
            lab[int(r["episode"])] = int(r["label"])
    eps = sorted(by_ep)
    return torch.tensor([sum(by_ep[e]) / len(by_ep[e]) for e in eps]), torch.tensor([lab[e] for e in eps])
