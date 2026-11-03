"""W2 D2: training stability tools."""
import math

import torch
import torch.nn as nn


def initial_mse_if_zero(y: torch.Tensor) -> float:
    """Expected MSE of a model that predicts 0 for every entry of y."""
    # TODO
    raise NotImplementedError


def initial_ce(num_classes: int) -> float:
    """Expected cross-entropy of a model that predicts the uniform distribution."""
    # TODO
    raise NotImplementedError


def warmup_cosine_lr(step: int, total: int, warmup: int, peak: float, final: float = 0.0) -> float:
    """Linear warmup 0 -> peak over `warmup` steps, then cosine from peak to `final` at `total`.
    Steps beyond `total` stay at `final`."""
    # TODO
    raise NotImplementedError


def clip_grad_norm(params: list[torch.Tensor], max_norm: float) -> float:
    """Clip the GLOBAL L2 norm of all .grad tensors to max_norm, in place.
    Params with grad None are skipped. Returns the norm BEFORE clipping (float).
    No torch.nn.utils."""
    # TODO
    raise NotImplementedError


def first_nonfinite_module(model: nn.Module, x: torch.Tensor) -> str | None:
    """Run model(x) with forward hooks on every LEAF module (no children).
    Return the name (from named_modules) of the first module, in execution order, whose output
    contains nan or inf; None if everything is finite. Remove all hooks before returning."""
    # TODO
    raise NotImplementedError
