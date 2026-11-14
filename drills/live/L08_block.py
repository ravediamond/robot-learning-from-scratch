"""L08 (45 min). A pre-norm transformer block, and its parameter count from the formula."""
import torch
from torch import nn


class Block(nn.Module):
    """x + attn(ln1(x)), then x + mlp(ln2(x)). ln1, ln2: nn.LayerNorm(dim). attn: nn.MultiheadAttention(dim, heads,
    batch_first=True) with a causal mask (bool mask, True = masked). mlp: Linear(dim, 4 dim) -> GELU -> Linear(4 dim, dim).
    Attribute names: ln1, attn, ln2, mlp."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        # TODO
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO
        raise NotImplementedError


def block_params(dim: int) -> int:
    """Parameter count of Block(dim, heads) from the formula (it does not depend on heads):
    attention 4 d^2 + 4 d, MLP 8 d^2 + 5 d, two LayerNorms 4 d."""
    # TODO
    raise NotImplementedError
