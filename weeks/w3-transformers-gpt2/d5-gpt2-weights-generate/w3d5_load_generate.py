"""W3 D5: load GPT-2 weights, generate, char-level data."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "d4-gpt2-model"))
from w3d4_gpt2 import GPT2, GPT2Config  # noqa: E402,F401

CONV1D = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")


def load_hf_state_dict(model: GPT2, hf_sd: dict[str, torch.Tensor]) -> int:
    """Copy a Hugging Face GPT2LMHeadModel state_dict into `model` (in place, no grad).
    Strip 'transformer.', transpose the CONV1D weights, skip keys ending in '.attn.bias' or
    '.attn.masked_bias'. lm_head.weight is tied: do not copy it twice.
    Raise ValueError if any of YOUR parameters is not filled or a shape differs.
    Return the number of tensors copied."""
    # TODO
    raise NotImplementedError


@torch.no_grad()
def generate(model: GPT2, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
             top_k: int | None = None, generator: torch.Generator | None = None) -> torch.Tensor:
    """idx (B, T0) -> (B, T0 + max_new_tokens). temperature == 0 -> greedy (argmax).
    Crop the context to model.cfg.n_positions. Use torch.multinomial(..., generator=generator)."""
    # TODO
    raise NotImplementedError


class CharTokenizer:
    """Vocabulary = sorted set of characters in `text`. encode: str -> list[int]; decode: inverse."""

    def __init__(self, text: str):
        # TODO
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        # TODO
        raise NotImplementedError

    def encode(self, s: str) -> list[int]:
        # TODO
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        # TODO
        raise NotImplementedError


def get_batch(data: torch.Tensor, block: int, batch: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """data (N,) long. Random starts i in [0, N - block - 1]; x = data[i:i+block], y = data[i+1:i+block+1].
    Returns x, y of shape (batch, block)."""
    # TODO
    raise NotImplementedError
