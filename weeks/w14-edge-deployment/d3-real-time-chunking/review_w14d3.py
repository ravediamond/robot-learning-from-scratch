"""REVIEW. Prefix used to condition the next chunk in the team's async runtime. With 150 ms inference the arm
visibly "backs up" a few millimetres at each chunk switch. Find the bug, fix, replace:
BUG:
"""
import torch


def prefix_target(prev_chunk: torch.Tensor, executed: int, delay_steps: int) -> torch.Tensor:
    """Known actions for the first delay_steps of the new chunk. executed = actions of prev_chunk already sent
    when the request was made."""
    return prev_chunk[:delay_steps]
