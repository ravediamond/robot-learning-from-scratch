import torch

from review_w7d4 import make_pairs


def test_review_next_frame():
    obs = torch.arange(2 * 4, dtype=torch.float32).view(2, 4, 1)      # value = (episode, t)
    act = torch.zeros(2, 3, 2)
    o_t, a_t, o_next = make_pairs(obs, act)
    assert torch.equal(o_next, o_t + 1), "the target must be the NEXT frame of the same episode"
