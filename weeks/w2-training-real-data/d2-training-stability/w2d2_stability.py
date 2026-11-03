"""W2 D2: SOLUTION."""
import math

import torch
import torch.nn as nn


def initial_mse_if_zero(y):
    return (y ** 2).mean().item()


def initial_ce(num_classes: int) -> float:
    return math.log(num_classes)


def warmup_cosine_lr(step, total, warmup, peak, final=0.0):
    if step < warmup:
        return peak * step / warmup
    if step >= total:
        return final
    p = (step - warmup) / (total - warmup)
    return final + (peak - final) * 0.5 * (1 + math.cos(math.pi * p))


def clip_grad_norm(params, max_norm):
    grads = [p.grad for p in params if p.grad is not None]
    total = torch.sqrt(sum((g.detach() ** 2).sum() for g in grads)).item()
    if total > max_norm:
        scale = max_norm / (total + 1e-6)
        for g in grads:
            g.mul_(scale)
    return total


def first_nonfinite_module(model, x):
    found: list[str] = []
    hooks = []
    for name, mod in model.named_modules():
        if len(list(mod.children())) == 0:
            def hook(m, inp, out, name=name):
                if not found and torch.is_tensor(out) and not torch.isfinite(out).all():
                    found.append(name)
            hooks.append(mod.register_forward_hook(hook))
    try:
        with torch.no_grad():
            model(x)
    finally:
        for h in hooks:
            h.remove()
    return found[0] if found else None
