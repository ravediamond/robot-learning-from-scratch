import torch

from review_w6d5 import dagger_step


def test_review_labels_come_from_the_expert():
    policy = lambda x: torch.zeros_like(x)
    expert = lambda x: torch.ones_like(x)
    O, A = torch.zeros(2, 2), torch.zeros(2, 2)
    O2, A2 = dagger_step(policy, expert, O, A, torch.randn(4, 2))
    assert torch.allclose(A2[2:], torch.ones(4, 2)), "visited states must be labelled by the expert"
