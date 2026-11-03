"""REVIEW. Executes one step of a policy chunk on the robot. The policy was trained on actions
divided by MAX_STEP (values around [-1, 1]). In the sim everything collides within a few steps,
"but the policy's val loss is excellent". Find the bug, fix, replace:
BUG:
"""
import torch

import sim2d


def execute_step(pos: torch.Tensor, chunk_scaled: torch.Tensor, k: int) -> torch.Tensor:
    """pos (N, 2), chunk_scaled (N, H, 2) = policy output in training units. Returns the new pos."""
    a = chunk_scaled[:, k]
    return pos + a
