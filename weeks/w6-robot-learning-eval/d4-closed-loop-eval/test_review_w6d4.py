import torch

import sim2d
from review_w6d4 import success_rate


def _start_dependent(pos, g):
    """Goes around the top only if it starts above the axis; otherwise straight into the obstacle."""
    up = sim2d.expert_action(pos, torch.ones(len(pos)))
    straight = sim2d.clip_action(sim2d.GOAL - pos)
    return torch.where(pos[:, 1:2] > 0, up, straight)


def test_review_same_starts_every_call():
    rates = {success_rate(_start_dependent, 100, seed=0) for _ in range(5)}
    assert len(rates) == 1, f"same seed, deterministic policy, different results: {sorted(rates)}"
