"""W1 D2: SOLUTION."""
import torch


def pairwise_sq_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    diff = a[:, None, :] - b[None, :, :]   # (N, 1, D) - (1, M, D) -> (N, M, D)
    return (diff ** 2).sum(dim=-1)         # (N, M)


def select_per_row(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return x.gather(dim=1, index=idx[:, None]).squeeze(1)   # (B, 1) -> (B,)


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask[..., None].to(x.dtype)            # (B, T, 1)
    total = (x * m).sum(dim=1)                 # (B, D)
    count = m.sum(dim=1).clamp(min=1.0)        # (B, 1): avoid 0/0 for empty samples
    return total / count                       # (B, D)


def batched_matvec(W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bij,bj->bi", W, x)    # (B, I)


def best_of_n(candidates: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    B = candidates.shape[0]
    best = scores.argmax(dim=1)                          # (B,)
    return candidates[torch.arange(B), best]             # (B, H, D)
