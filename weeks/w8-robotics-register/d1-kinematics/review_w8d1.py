"""REVIEW. SOLUTION.

BUG: the matrix is the TRANSPOSE of Ry(t) (= Ry(-t)): every pitch joint turns the wrong way. It is still a
valid rotation (orthonormal, det 1), so nothing crashes and poses look plausible. Check: Ry(+90) x = -z.
"""
import torch


def rot_y(t):
    c, s = torch.cos(t), torch.sin(t)
    one, zero = torch.ones_like(t), torch.zeros_like(t)
    return torch.stack([torch.stack([c, zero, s]), torch.stack([zero, one, zero]), torch.stack([-s, zero, c])])
