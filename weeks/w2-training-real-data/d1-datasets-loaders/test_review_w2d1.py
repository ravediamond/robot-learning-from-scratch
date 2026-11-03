import torch

from review_w2d1 import window_actions


def test_review_window_stays_in_episode():
    ep = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    act = (ep * 100 + torch.arange(7))[:, None].float()        # (7, 1)
    w, mask = window_actions(act, ep, t=1, H=4)                  # episode 0 ends at frame 2
    assert w[:, 0].tolist() == [1.0, 2.0, 2.0, 2.0], f"window leaked into the next episode: {w[:, 0].tolist()}"
    assert mask.tolist() == [True, True, False, False]
    w, mask = window_actions(act, ep, t=5, H=4)
    assert w[:, 0].tolist() == [105.0, 106.0, 106.0, 106.0] and mask.sum() == 2
