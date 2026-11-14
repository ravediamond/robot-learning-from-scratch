"""L05 (30 min). Rotary position embeddings."""
import torch


def rope_cache(T: int, D: int, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """cos, sin of shape (T, D/2): angle[t, i] = t * base ** (-2i / D)."""
    # TODO
    raise NotImplementedError


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x (..., T, D). Pair dimension i with i + D/2 ("rotate half" convention):
    out[:i] = x1 * cos - x2 * sin, out[D/2 + i] = x1 * sin + x2 * cos, with x1 = x[..., :D/2], x2 = x[..., D/2:]."""
    # TODO
    raise NotImplementedError
