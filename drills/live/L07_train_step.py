"""L07 (45 min). The training loop people get wrong: accumulation, clipping, warmup + cosine."""
import math

import torch


def lr_at(step: int, base_lr: float, warmup: int, total: int, min_lr: float = 0.0) -> float:
    """Linear warmup from 0 over `warmup` steps (lr = base * (step + 1) / warmup for step < warmup), then cosine
    from base_lr to min_lr over the remaining steps: progress = (step - warmup) / max(1, total - warmup),
    lr = min_lr + 0.5 * (base - min_lr) * (1 + cos(pi * min(progress, 1)))."""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = min((step - warmup) / max(1, total - warmup), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def train(model: torch.nn.Module, batches: list[tuple[torch.Tensor, torch.Tensor]], accum: int, base_lr: float,
          warmup: int, total_updates: int, max_norm: float) -> list[float]:
    """SGD (torch.optim.SGD, lr set every update from lr_at(update index, ...)). Micro-batches come in order;
    every `accum` micro-batches make ONE update whose gradient equals the gradient of the mean loss over those
    micro-batches (loss = MSE of model(x) vs y, divided by accum before backward). Clip the global gradient norm
    to max_norm (torch.nn.utils.clip_grad_norm_) after accumulating, before the step. Zero the gradients after
    each step. Return the list of the learning rates used, one per update. Leftover micro-batches are ignored."""
    opt = torch.optim.SGD(model.parameters(), lr=base_lr)
    lrs = []
    n_updates = len(batches) // accum
    for u in range(n_updates):
        for x, y in batches[u * accum:(u + 1) * accum]:
            loss = (model(x) - y).pow(2).mean() / accum
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        lr = lr_at(u, base_lr, warmup, total_updates)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.step()
        opt.zero_grad()
        lrs.append(lr)
    return lrs
