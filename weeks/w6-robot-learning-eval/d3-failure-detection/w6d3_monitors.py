"""W6 D3: failure-monitor metrics. Labels: 1 = failure (positive), 0 = success. Higher score = more alarming."""
import math

import torch


def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Rank-based AUROC with average ranks for ties. Needs both classes present."""
    # TODO
    raise NotImplementedError


def auroc_ci(scores: torch.Tensor, labels: torch.Tensor, B: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap over episodes (rows). Skip resamples that contain a single class."""
    # TODO
    raise NotImplementedError


def tpr_at_fpr(scores: torch.Tensor, labels: torch.Tensor, fpr: float) -> float:
    """Threshold = torch.quantile(success scores, 1 - fpr). Flag score > threshold. Return the fraction of
    failures flagged."""
    # TODO
    raise NotImplementedError


def conformal_threshold(calib_success_scores: torch.Tensor, alpha: float) -> float:
    """k = ceil((n + 1)(1 - alpha)); return the k-th smallest calibration score (1-based).
    If k > n return float('inf') (no finite threshold gives the guarantee)."""
    # TODO
    raise NotImplementedError
