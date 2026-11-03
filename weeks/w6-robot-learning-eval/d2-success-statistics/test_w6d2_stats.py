import torch

import w6d2_stats as m


def test_wilson():
    lo, hi = m.wilson(7, 10)
    assert abs(lo - 0.3968) < 1e-3 and abs(hi - 0.8922) < 1e-3
    assert m.wilson(0, 10)[0] == 0.0 and abs(m.wilson(0, 10)[1] - 0.2775) < 1e-3
    assert m.wilson(10, 10)[1] == 1.0 and m.wilson(0, 0) == (0.0, 1.0)


def test_bootstrap():
    g = torch.Generator().manual_seed(0)
    x = (torch.rand(40, generator=g) < 0.6).float()
    lo, hi = m.bootstrap_ci(x, lambda v: v.mean(), 2000, seed=0)
    assert lo < x.mean().item() < hi and 0.1 < hi - lo < 0.45
    assert m.bootstrap_ci(x, lambda v: v.mean(), 2000, seed=0) == (lo, hi), "seeded = reproducible"
    big = x.repeat(25)
    lo2, hi2 = m.bootstrap_ci(big, lambda v: v.mean(), 2000, seed=0)
    assert hi2 - lo2 < 0.4 * (hi - lo), "more episodes -> narrower"


def test_paired():
    a = torch.tensor([1, 1, 1, 0, 1, 1, 0, 1, 1, 1.0])
    b = torch.tensor([1, 0, 1, 0, 0, 1, 0, 1, 0, 1.0])
    d, lo, hi = m.paired_bootstrap_diff(a, b)
    assert abs(d - 0.3) < 1e-6 and lo <= d <= hi and lo >= 0.0


def test_mcnemar():
    assert abs(m.mcnemar_exact(5, 1) - 0.21875) < 1e-9
    assert abs(m.mcnemar_exact(0, 0) - 1.0) < 1e-9
    assert m.mcnemar_exact(10, 0) < 0.002


def test_trials_needed():
    assert m.trials_needed(0.6, 0.8) == 82
    assert m.trials_needed(0.8, 0.6) == 82
    assert m.trials_needed(0.5, 0.9) < m.trials_needed(0.6, 0.8)
