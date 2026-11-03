"""REVIEW. SOLUTION.

BUG: the policy output is in training units (divided by MAX_STEP) and is applied raw: steps are 20x too
large. Un-normalize (x MAX_STEP) and clip every command before execution.
"""
import torch

import sim2d


def execute_step(pos, chunk_scaled, k):
    return pos + sim2d.clip_action(chunk_scaled[:, k] * sim2d.MAX_STEP)
