"""W5 D4: LoRA."""
import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps an nn.Linear. self.base (frozen: requires_grad False on its weight and bias),
    self.A (r, in) ~ N(0, 1/r), self.B (out, r) zeros, self.scale = alpha / r.
    forward(x) = base(x) + scale * (x @ A.T) @ B.T"""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        # TODO

    def forward(self, x):
        # TODO
        raise NotImplementedError

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        """A new nn.Linear with weight W + scale * B @ A (same bias)."""
        # TODO
        raise NotImplementedError


def apply_lora(model: nn.Module, names: list[str], r: int = 8, alpha: float = 16.0) -> nn.Module:
    """Freeze EVERY parameter of model, then replace each submodule named in `names` (dotted names as in
    named_modules, e.g. 'blocks.1.attn.q_proj', each an nn.Linear) by LoRALinear(module, r, alpha).
    In place; returns model."""
    # TODO
    raise NotImplementedError


def trainable_params(model: nn.Module) -> int:
    # TODO
    raise NotImplementedError
