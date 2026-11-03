"""REVIEW. CFG for an instruction-conditioned action head. Raising the guidance weight was
supposed to make the robot follow the instruction more strictly. w=1 behaves like the
unconditional model, and w=0 follows instructions. Find the bug, fix, replace:
BUG:
"""
import torch


def guided_velocity(model, x, t, cond, null_cond, w: float) -> torch.Tensor:
    v_c = model(x, t, cond)
    v_u = model(x, t, null_cond)
    return (1 - w) * v_c + w * v_u
