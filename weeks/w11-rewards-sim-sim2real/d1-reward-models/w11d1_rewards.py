"""W11 D1: success classifier, precision-first threshold, pairwise progress model."""
import torch
from torch import nn


def mlp(inp: int, out: int = 1, hidden: int = 64) -> nn.Module:
    return nn.Sequential(nn.Linear(inp, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out))


def train_success_classifier(x: torch.Tensor, y: torch.Tensor, steps: int = 800, batch: int = 256,
                             lr: float = 3e-3, seed: int = 0) -> nn.Module:
    """mlp(x.shape[1]) trained with binary cross-entropy on logits. Return the model (outputs LOGITS (B, 1))."""
    # TODO
    raise NotImplementedError


def threshold_for_precision(scores: torch.Tensor, labels: torch.Tensor, min_precision: float) -> float:
    """Smallest threshold t (taken among the scores) such that predicting positive when score >= t has
    precision >= min_precision. Smallest = highest recall. If none qualifies, return float('inf')."""
    # TODO
    raise NotImplementedError


def progress_pairs(episode: torch.Tensor, t: torch.Tensor, n_pairs: int, generator=None):
    """Sample n_pairs index pairs (a, b) of rows that belong to the SAME episode and have different t.
    Return (idx_a, idx_b, label) with label = 1.0 if t[b] > t[a] else 0.0.
    Assume every episode has the same length T and rows are stored time-major (row = t * n_episodes + e),
    as in reward_data.expert_episodes."""
    # TODO
    raise NotImplementedError


def pairwise_loss(score_a: torch.Tensor, score_b: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """Bradley-Terry: binary cross-entropy with logits on (score_b - score_a) against label."""
    # TODO
    raise NotImplementedError


def train_progress_model(x: torch.Tensor, episode: torch.Tensor, t: torch.Tensor, steps: int = 800,
                         batch: int = 256, lr: float = 3e-3, seed: int = 0) -> nn.Module:
    """mlp(x.shape[1]) -> scalar score, trained with pairwise_loss on progress_pairs. Return the model."""
    # TODO
    raise NotImplementedError


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman rank correlation = Pearson correlation of the ranks (ties: any consistent order is fine)."""
    # TODO
    raise NotImplementedError
