"""REVIEW. Checkpoint identity check in the eval launcher: it refuses to run if the weights' hash differs from
the one in the experiment register. Two different fine-tuned checkpoints passed as "the same model" and the
table compared a policy with itself. Find the bug, fix, replace:
BUG:
"""
import hashlib

import torch


def weights_hash(state_dict: dict[str, torch.Tensor]) -> str:
    return hashlib.sha256(str(state_dict).encode()).hexdigest()
