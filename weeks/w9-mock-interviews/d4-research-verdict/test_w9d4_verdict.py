import torch

import w9d4_verdict as m
from mock_experiment import FAIL_LINE, PASS_LINE, mock_experiment


def test_rule():
    assert m.verdict(0.20, 0.08, 0.32, 0.10, 0.02) == "pass"
    assert m.verdict(0.20, -0.01, 0.40, 0.10, 0.02) == "not proven", "CI includes 0: cannot tell"
    assert m.verdict(0.00, -0.05, 0.05, 0.10, 0.02) == "fail"
    assert m.verdict(0.00, -0.15, 0.15, 0.10, 0.02) == "not proven", "too wide to rule out the pass line"
    assert m.verdict(0.06, 0.02, 0.09, 0.10, 0.02) == "not proven", "between the lines"
    assert m.verdict(0.20, 0.08, 0.32, 0.10, 0.02, ["policy mix"]) == "not proven"


def test_paired_ci():
    b = torch.tensor([0, 1, 0, 1, 1, 0, 0, 1.0])
    a = torch.tensor([1, 1, 0, 1, 1, 1, 0, 1.0])
    d, lo, hi = m.paired_diff_ci(b, a, 2000, 0)
    assert abs(d - 0.25) < 1e-9 and lo <= d <= hi and lo >= 0


def test_mock():
    base, method = mock_experiment()
    d, lo, hi = m.paired_diff_ci(base, method)
    v = m.verdict(d, lo, hi, PASS_LINE, FAIL_LINE)
    print(f"\nmock: d = {d:+.3f}  CI [{lo:+.3f}, {hi:+.3f}]  ->  {v}")
    assert v == "not proven"
