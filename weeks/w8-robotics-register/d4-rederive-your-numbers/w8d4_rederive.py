"""W8 D4: re-derive your own published numbers from the raw files in reference/."""
import csv
import json
import math
from pathlib import Path

import torch


def wilson(k: int, n: int, z: float) -> tuple[float, float]:
    # TODO
    raise NotImplementedError


def audit_counts(path: str | Path, z: float, tol: float) -> list[str]:
    """counts_ci.json: {name: {"k", "n", "wilson95": [lo, hi], ...}}. Return the names whose stored Wilson
    bounds differ from yours by more than tol (either bound), in file order."""
    # TODO
    raise NotImplementedError


def load_anchor(path: str | Path) -> dict:
    """anchor_predictions.csv (episode, label, feature, seed, score) ->
    {feature: {"episodes": sorted episode ids (E,) long, "labels": (E,) long, "scores": (n_seeds, E) float64}}."""
    # TODO
    raise NotImplementedError


def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Rank AUROC with average ranks for ties (label 1 = failure = positive)."""
    # TODO
    raise NotImplementedError


def recompute_anchor(path: str | Path) -> dict:
    """{feature: {"per_seed": [AUROC per seed], "mean_of_seeds": mean of those,
                  "seed_averaged": AUROC of the scores averaged over seeds}}."""
    # TODO
    raise NotImplementedError


def bootstrap_auroc_ci(scores: torch.Tensor, labels: torch.Tensor, B: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Episode bootstrap, percentile 95%, resamples with a single class skipped, torch.Generator(seed)."""
    # TODO
    raise NotImplementedError
