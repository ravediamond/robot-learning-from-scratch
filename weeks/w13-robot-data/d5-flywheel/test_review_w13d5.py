import numpy as np

from review_w13d5 import recovery_samples


def test_review_labels_are_human_actions():
    log = {"obs": np.zeros((5, 2)), "executed_action": np.ones((5, 2)), "policy_action": -np.ones((5, 2)),
           "intervention": np.array([0, 1, 1, 0, 0])}
    s = recovery_samples(log)
    assert len(s) == 2 and all((a == 1).all() for _, a in s)
