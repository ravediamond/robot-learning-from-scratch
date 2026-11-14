"""L03 (30 min). Cross-entropy from logits, numerically stable, with ignore_index and label smoothing."""
import torch


def cross_entropy(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = -100,
                  label_smoothing: float = 0.0) -> torch.Tensor:
    """logits (N, K), target (N,) int. Mean over the non-ignored rows of
    -sum_k q_k log p_k, with q = (1 - eps) * onehot(target) + eps / K and log p from a stable log-softmax
    (subtract the row max; use logsumexp). Do not call F.cross_entropy or F.log_softmax.
    If every row is ignored, return 0.0 (as a tensor) instead of NaN."""
    # TODO
    raise NotImplementedError
