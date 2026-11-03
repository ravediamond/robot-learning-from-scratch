import torch

from review_w6d3 import fast_auroc


def test_review_ties_and_order():
    scores = torch.tensor([3.0, 3.0, 3.0, 3.0, 5.0, 1.0])
    labels = torch.tensor([1, 0, 1, 0, 1, 0])
    a = fast_auroc(scores, labels)
    perm = torch.tensor([5, 3, 1, 4, 2, 0])
    b = fast_auroc(scores[perm], labels[perm])
    assert abs(a - b) < 1e-9, "AUROC changed with the row order"
    assert abs(a - 7 / 9) < 1e-6, f"ties must count 1/2: expected 0.778, got {a:.3f}"
