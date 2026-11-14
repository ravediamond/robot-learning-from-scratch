"""REVIEW. SOLUTION.

BUG: the prefix is taken from the START of the previous chunk, which was executed long ago, so the new chunk is
pulled back to old positions: the arm backs up at every switch. The actions the robot will execute during
inference are the ones right after those already sent: prev_chunk[executed : executed + delay_steps].
"""
import torch


def prefix_target(prev_chunk: torch.Tensor, executed: int, delay_steps: int) -> torch.Tensor:
    return prev_chunk[executed:executed + delay_steps]
