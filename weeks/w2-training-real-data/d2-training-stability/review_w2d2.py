"""REVIEW. SOLUTION.

BUG: clipping runs before backward(): the grads are still None (just zeroed), so it clips nothing;
backward() then writes the huge gradient and step() applies it. Clip between backward() and step().
"""
import torch


def train_step(model, opt, loss_fn, x, y, max_norm: float = 1.0) -> float:
    opt.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    opt.step()
    return loss.item()
