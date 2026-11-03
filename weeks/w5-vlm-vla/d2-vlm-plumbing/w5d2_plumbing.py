"""W5 D2: VLM plumbing."""
import torch
import torch.nn as nn


def pixel_shuffle_tokens(x: torch.Tensor, r: int) -> torch.Tensor:
    """x (B, N, C) tokens of a square h x h grid (row-major), h divisible by r.
    Returns (B, N / r^2, C * r^2): token (i, j) of the new (h/r x h/r) grid concatenates the r x r block
    of old tokens [r*i .. r*i+r-1] x [r*j .. r*j+r-1], ordered (row in block, col in block, channel)."""
    # TODO
    raise NotImplementedError


class Projector(nn.Module):
    """Linear(in_dim, out_dim) -> GELU -> Linear(out_dim, out_dim)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        # TODO

    def forward(self, x):
        # TODO
        raise NotImplementedError


def block_attention_mask(block_sizes: list[int], causal_last: bool = False) -> torch.Tensor:
    """(T, T) bool, True = query may attend key. A token in block b attends every token in blocks <= b
    (bidirectional inside its block). If causal_last, the LAST block is causal inside itself."""
    # TODO
    raise NotImplementedError


def assemble_prefix(img_tokens: list[torch.Tensor], txt_tokens: torch.Tensor, txt_mask: torch.Tensor,
                    state_token: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """img_tokens: list of (B, Ni, D) (one per camera), txt_tokens (B, Nt, D), txt_mask (B, Nt) bool
    (False = padding), state_token (B, 1, D).
    Returns seq (B, T, D) = [cam_0, cam_1, ..., text, state] and key_mask (B, T) bool (False on text padding)."""
    # TODO
    raise NotImplementedError
