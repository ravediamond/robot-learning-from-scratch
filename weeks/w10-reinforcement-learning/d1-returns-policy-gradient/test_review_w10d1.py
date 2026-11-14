import torch

from review_w10d1 import returns


def test_review_no_leak_across_episodes():
    r = torch.tensor([[1.0], [1.0], [100.0], [100.0]])
    d = torch.tensor([[False], [True], [False], [True]])
    G = returns(r, d, 1.0)
    assert G.flatten().tolist() == [2.0, 1.0, 200.0, 100.0]
