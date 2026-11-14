import torch

from review_w10d5 import label


def test_review_each_task_gets_positives():
    g = torch.Generator().manual_seed(0)
    adv = torch.cat([torch.randn(500, generator=g) * 0.5, torch.randn(500, generator=g) * 8])
    task = torch.cat([torch.zeros(500), torch.ones(500)]).long()
    ind = label(adv, task)
    for t in (0, 1):
        frac = ind[task == t].float().mean().item()
        assert abs(frac - 0.3) < 0.03, f"task {t}: {frac:.2f} positives, expected 0.30"
