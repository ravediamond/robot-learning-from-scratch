import torch

from review_w1d5 import make_val_mask


def test_review_no_episode_in_both_splits():
    lengths = torch.randint(50, 150, (40,), generator=torch.Generator().manual_seed(1))
    ep = torch.repeat_interleave(torch.arange(40), lengths)
    val = make_val_mask(ep, 0.2, seed=0)
    val_eps = set(ep[val].tolist())
    train_eps = set(ep[~val].tolist())
    assert val_eps.isdisjoint(train_eps), f"{len(val_eps & train_eps)} episodes appear in both splits"
    assert 4 <= len(val_eps) <= 12, "about 20% of episodes should be in val"
