"""L03 (30 min). Cross-entropy from logits, numerically stable, with ignore_index and label smoothing."""
import torch


def cross_entropy(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = -100,
                  label_smoothing: float = 0.0) -> torch.Tensor:
    """logits (N, K), target (N,) int. Mean over the non-ignored rows of
    -sum_k q_k log p_k, with q = (1 - eps) * onehot(target) + eps / K and log p from a stable log-softmax
    (subtract the row max; use logsumexp). Do not call F.cross_entropy or F.log_softmax.
    If every row is ignored, return 0.0 (as a tensor) instead of NaN."""
    keep = target != ignore_index
    if not keep.any():
        return logits.sum() * 0.0
    z = logits[keep]
    t = target[keep]
    z = z - z.max(dim=-1, keepdim=True).values
    logp = z - torch.logsumexp(z, dim=-1, keepdim=True)
    K = z.shape[-1]
    nll = -logp.gather(1, t[:, None]).squeeze(1)
    smooth = -logp.mean(dim=-1)
    return ((1 - label_smoothing) * nll + label_smoothing * smooth).mean()
