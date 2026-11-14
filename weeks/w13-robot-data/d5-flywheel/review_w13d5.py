"""REVIEW. Builds the "recovery" dataset from intervention logs for the next training round. After retraining,
the policy fails MORE often in exactly the situations where operators had to take over.
Find the bug, fix, replace:
BUG:
"""
import numpy as np


def recovery_samples(log: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    """log: {"obs", "executed_action", "policy_action", "intervention"} arrays over time."""
    idx = np.where(log["intervention"] == 1)[0]
    return [(log["obs"][t], log["policy_action"][t]) for t in idx]
