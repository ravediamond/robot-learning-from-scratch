"""W11 D1: success classifier, precision-first threshold, pairwise progress model."""
import torch
from torch import nn


def mlp(inp: int, out: int = 1, hidden: int = 64) -> nn.Module:
    return nn.Sequential(nn.Linear(inp, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out))


def train_success_classifier(x: torch.Tensor, y: torch.Tensor, steps: int = 800, batch: int = 256,
                             lr: float = 3e-3, seed: int = 0) -> nn.Module:
    """mlp(x.shape[1]) trained with binary cross-entropy on logits. Return the model (outputs LOGITS (B, 1))."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = mlp(x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        idx = torch.randint(0, len(x), (batch,), generator=g)
        loss = nn.functional.binary_cross_entropy_with_logits(model(x[idx]).squeeze(-1), y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def threshold_for_precision(scores: torch.Tensor, labels: torch.Tensor, min_precision: float) -> float:
    """Smallest threshold t (taken among the scores) such that predicting positive when score >= t has
    precision >= min_precision. Smallest = highest recall. If none qualifies, return float('inf')."""
    order = torch.argsort(scores, descending=True)
    s, l = scores[order], labels[order].float()
    tp = torch.cumsum(l, dim=0)
    precision = tp / torch.arange(1, len(s) + 1)
    # a threshold must include every sample with the same score: keep only the last index of each tie group
    last_of_group = torch.ones(len(s), dtype=torch.bool)
    last_of_group[:-1] = s[:-1] != s[1:]
    ok = (precision >= min_precision) & last_of_group
    if not ok.any():
        return float("inf")
    return s[torch.nonzero(ok).max()].item()


def progress_pairs(episode: torch.Tensor, t: torch.Tensor, n_pairs: int, generator=None):
    """Sample n_pairs index pairs (a, b) of rows that belong to the SAME episode and have different t.
    Return (idx_a, idx_b, label) with label = 1.0 if t[b] > t[a] else 0.0.
    Assume every episode has the same length T and rows are stored time-major (row = t * n_episodes + e),
    as in reward_data.expert_episodes."""
    n_ep = int(episode.max().item()) + 1
    T = len(episode) // n_ep
    e = torch.randint(0, n_ep, (n_pairs,), generator=generator)
    ta = torch.randint(0, T, (n_pairs,), generator=generator)
    tb = (ta + torch.randint(1, T, (n_pairs,), generator=generator)) % T  # never equal to ta
    idx_a, idx_b = ta * n_ep + e, tb * n_ep + e
    return idx_a, idx_b, (t[idx_b] > t[idx_a]).float()


def pairwise_loss(score_a: torch.Tensor, score_b: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """Bradley-Terry: binary cross-entropy with logits on (score_b - score_a) against label."""
    return nn.functional.binary_cross_entropy_with_logits(score_b - score_a, label)


def train_progress_model(x: torch.Tensor, episode: torch.Tensor, t: torch.Tensor, steps: int = 800,
                         batch: int = 256, lr: float = 3e-3, seed: int = 0) -> nn.Module:
    """mlp(x.shape[1]) -> scalar score, trained with pairwise_loss on progress_pairs. Return the model."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = mlp(x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        ia, ib, y = progress_pairs(episode, t, batch, g)
        loss = pairwise_loss(model(x[ia]).squeeze(-1), model(x[ib]).squeeze(-1), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman rank correlation = Pearson correlation of the ranks (ties: any consistent order is fine)."""
    ra = torch.argsort(torch.argsort(a)).float()
    rb = torch.argsort(torch.argsort(b)).float()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return ((ra * rb).sum() / (ra.norm() * rb.norm())).item()
