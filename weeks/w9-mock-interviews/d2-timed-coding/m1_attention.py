# time:
import torch


def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """(B, h, T, d) each -> (B, h, T, d). Scaled dot-product, causal."""
    raise NotImplementedError


def multi_head_self_attention(x: torch.Tensor, Wqkv: torch.Tensor, Wo: torch.Tensor, h: int) -> torch.Tensor:
    """x (B, T, D), Wqkv (3D, D) applied as x @ Wqkv.T then split into q, k, v; Wo (D, D) applied as out @ Wo.T.
    Causal. Returns (B, T, D)."""
    raise NotImplementedError
