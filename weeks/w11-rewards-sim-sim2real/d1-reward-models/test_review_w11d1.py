import torch

from review_w11d1 import sample_pairs


def test_review_pairs_within_episode():
    n_ep, T = 25, 12
    episode = torch.arange(n_ep).repeat(T)
    t = torch.arange(T).repeat_interleave(n_ep)
    ia, ib, y = sample_pairs(episode, t, 1000, torch.Generator().manual_seed(0))
    assert len(ia) > 500
    assert torch.equal(episode[ia], episode[ib]), "a pair must come from one episode"
    assert (t[ia] != t[ib]).all()
