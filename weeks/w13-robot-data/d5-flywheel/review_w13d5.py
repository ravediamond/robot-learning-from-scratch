"""REVIEW. SOLUTION.

BUG: the samples are labelled with the policy's proposed actions, which are exactly the actions the human
overrode. Training on them teaches the policy its own mistakes in the states where it fails. Label with the
executed (human) actions recorded during the takeover.
"""
import numpy as np


def recovery_samples(log: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    idx = np.where(log["intervention"] == 1)[0]
    return [(log["obs"][t], log["executed_action"][t]) for t in idx]
