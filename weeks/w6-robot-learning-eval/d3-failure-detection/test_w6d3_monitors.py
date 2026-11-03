import itertools

import torch

import w6d3_monitors as m


def _brute(scores, labels):
    pos, neg = scores[labels == 1], scores[labels == 0]
    tot = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p, n in itertools.product(pos.tolist(), neg.tolist()))
    return tot / (len(pos) * len(neg))


def test_auroc_matches_pairs_with_ties():
    g = torch.Generator().manual_seed(0)
    scores = torch.randint(0, 5, (60,), generator=g).float()          # many ties, like episode lengths
    labels = (torch.rand(60, generator=g) < 0.4).long()
    assert abs(m.auroc(scores, labels) - _brute(scores, labels)) < 1e-9
    assert m.auroc(torch.ones(10), torch.tensor([1, 0] * 5)) == 0.5


def test_auroc_ci():
    g = torch.Generator().manual_seed(1)
    lab = torch.tensor([1] * 20 + [0] * 40)
    s = torch.cat([torch.randn(20, generator=g) + 1.2, torch.randn(40, generator=g)])
    a = m.auroc(s, lab)
    lo, hi = m.auroc_ci(s, lab, 1000, seed=0)
    assert lo < a < hi and 0.1 < hi - lo < 0.4
    few = m.auroc_ci(torch.cat([s[:2], s[20:24]]), torch.tensor([1, 1, 0, 0, 0, 0]), 1000, seed=0)
    assert few[0] <= few[1], "single-class resamples must be skipped, not crash"


def test_tpr_at_fpr():
    succ = torch.arange(100.0)
    fail = torch.tensor([50.0, 95.0, 99.5, 200.0])
    s, l = torch.cat([succ, fail]), torch.tensor([0] * 100 + [1] * 4)
    assert abs(m.tpr_at_fpr(s, l, 0.05) - 0.75) < 1e-9         # threshold 94.05: flags 95, 99.5, 200


def test_conformal_guarantee():
    assert m.conformal_threshold(torch.arange(1.0, 20.0), 0.1) == 18.0     # n=19, k=ceil(20*0.9)=18
    assert m.conformal_threshold(torch.arange(5.0), 0.1) == float("inf")
    g = torch.Generator().manual_seed(0)
    alarms = []
    for _ in range(400):
        calib, new = torch.randn(30, generator=g), torch.randn(50, generator=g)
        alarms.append((new > m.conformal_threshold(calib, 0.1)).float().mean())
    rate = torch.stack(alarms).mean().item()
    assert rate <= 0.1 + 0.01, f"false-alarm rate {rate:.3f} above alpha"
