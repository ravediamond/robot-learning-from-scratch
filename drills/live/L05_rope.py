"""L05 (30 min). Rotary position embeddings."""
import torch


def rope_cache(T: int, D: int, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """cos, sin of shape (T, D/2): angle[t, i] = t * base ** (-2i / D)."""
    inv = base ** (-torch.arange(0, D, 2, dtype=torch.float32) / D)      # (D/2,)
    ang = torch.arange(T, dtype=torch.float32)[:, None] * inv[None]     # (T, D/2)
    return ang.cos(), ang.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x (..., T, D). Pair dimension i with i + D/2 ("rotate half" convention):
    out[:i] = x1 * cos - x2 * sin, out[D/2 + i] = x1 * sin + x2 * cos, with x1 = x[..., :D/2], x2 = x[..., D/2:]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
