"""W5 D5: a mini VLA. Convention: t=0 noise, t=1 data. Chunks in training units ([-1, 1])."""
import torch
import torch.nn as nn

from vla_blocks import Block, time_embedding


class MiniVLA(nn.Module):
    """See the README for the exact layout. Defaults: D=64, H=4, heads=4, patch=4, img_size=16, vocab=6."""

    def __init__(self, D: int = 64, H: int = 4, heads: int = 4, patch: int = 4, img_size: int = 16, vocab: int = 6):
        super().__init__()
        # TODO

    def encode_prefix(self, img, tokens, state) -> torch.Tensor:
        """(B,3,S,S), (B,4) long, (B,2) -> prefix features (B, n_patch + 4 + 1, D)."""
        # TODO
        raise NotImplementedError

    def expert(self, prefix, x_t, t) -> torch.Tensor:
        """prefix (B, P, D), x_t (B, H, 2), t (B,) -> velocity (B, H, 2)."""
        # TODO
        raise NotImplementedError

    def forward(self, img, tokens, state, x_t, t):
        # TODO
        raise NotImplementedError

    @torch.no_grad()
    def act(self, img, tokens, state, generator=None, steps: int = 10) -> torch.Tensor:
        """Chunk (B, H, 2) in training units. encode_prefix exactly ONCE, then `steps` Euler steps."""
        # TODO
        raise NotImplementedError


def train_vla(demos: dict, steps: int = 2000, batch: int = 128, lr: float = 1e-3, seed: int = 0) -> MiniVLA:
    """AdamW (weight decay 0.01), linear warmup over 200 steps, grad clip 1.0, flow-matching loss.
    Return in eval mode."""
    # TODO
    raise NotImplementedError
