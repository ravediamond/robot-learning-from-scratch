"""REVIEW. SOLUTION.

BUG: one global threshold. Advantage scales differ per task, so the top 30% overall comes almost entirely
from one task and the other has (nearly) no positive labels: the bit encodes the task, not the quality.
Compute the quantile within each task.
"""
import torch


def label(adv: torch.Tensor, task_id: torch.Tensor, q: float = 0.7) -> torch.Tensor:
    out = torch.zeros_like(adv, dtype=torch.bool)
    for t in task_id.unique():
        m = task_id == t
        out[m] = adv[m] > torch.quantile(adv[m], q)
    return out
