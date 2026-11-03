"""W3 D4: GPT-2, with the exact module names of the original (so weights can be loaded)."""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPT2Config:
    vocab_size: int = 50257
    n_positions: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12


class CausalSelfAttention(nn.Module):
    """Attributes c_attn = Linear(D, 3D), c_proj = Linear(D, D). (B, T, D) -> (B, T, D), causal."""

    def __init__(self, cfg: GPT2Config):
        super().__init__()
        # TODO

    def forward(self, x):
        # TODO
        raise NotImplementedError


class MLP(nn.Module):
    """c_fc = Linear(D, 4D), GELU(tanh), c_proj = Linear(4D, D)."""

    def __init__(self, cfg: GPT2Config):
        super().__init__()
        # TODO

    def forward(self, x):
        # TODO
        raise NotImplementedError


class GPT2Block(nn.Module):
    """ln_1, attn, ln_2, mlp; pre-norm with residuals. LayerNorm eps 1e-5."""

    def __init__(self, cfg: GPT2Config):
        super().__init__()
        # TODO

    def forward(self, x):
        # TODO
        raise NotImplementedError


class GPT2(nn.Module):
    """Attributes: wte, wpe (nn.Embedding), h (nn.ModuleList of GPT2Block), ln_f, lm_head
    (Linear(D, V, bias=False) whose weight IS wte.weight). Init as in the README."""

    def __init__(self, cfg: GPT2Config):
        super().__init__()
        # TODO

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """idx (B, T) long, T <= n_positions. Returns (logits (B, T, V), loss or None)."""
        # TODO
        raise NotImplementedError
