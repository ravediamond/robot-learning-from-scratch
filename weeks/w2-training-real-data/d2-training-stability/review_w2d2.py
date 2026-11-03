"""REVIEW. A training step with gradient clipping, added after a teleop glitch produced a
loss spike that destroyed the policy. The spike happened again. Find the bug, fix, replace:
BUG:
"""
import torch


def train_step(model, opt, loss_fn, x, y, max_norm: float = 1.0) -> float:
    opt.zero_grad()
    loss = loss_fn(model(x), y)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    loss.backward()
    opt.step()
    return loss.item()
