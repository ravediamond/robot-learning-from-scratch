import numpy as np
import torch

from w14d5_rollout import (canary_decision, fingerprint, rate_ratio_pvalue, shadow_disagreement, stage_sizes,
                           verify_artifact)


def test_verify():
    sd = {"w": torch.ones(3)}
    fp = fingerprint(sd)
    assert verify_artifact(sd, fp) and not verify_artifact({"w": torch.zeros(3)}, fp)


def test_shadow():
    dep = np.zeros((5, 2))
    cand = np.array([[0, 0], [0.2, 0], [0, -0.05], [0, 0.11], [0, 0]])
    r = shadow_disagreement(dep, cand, 0.1)
    assert r == {"rate": 0.4, "steps": [1, 3], "max_abs": 0.2}


def test_rate_test():
    assert rate_ratio_pvalue(0, 10, 0, 10) == 1.0
    assert abs(rate_ratio_pvalue(3, 40, 30, 400) - 0.5875) < 1e-3
    assert rate_ratio_pvalue(12, 40, 30, 400) < 1e-3
    assert abs(rate_ratio_pvalue(1, 1, 0, 1) - 0.5) < 1e-12


def test_decision():
    assert canary_decision(3, 10, 30, 400) == "wait"
    assert canary_decision(3, 40, 30, 400) == "continue"
    assert canary_decision(12, 40, 30, 400) == "rollback"
    assert canary_decision(5, 40, 30, 400) == "continue", "5 vs an expected 3: not evidence yet"


def test_stages():
    assert stage_sizes(50, [0.05, 0.25, 1.0]) == [3, 13, 50]
    assert stage_sizes(10, [0.01, 0.05, 0.5, 1.0]) == [1, 1, 5, 10]
