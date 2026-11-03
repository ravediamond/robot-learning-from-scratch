import torch

from review_w6d2 import success_ci


def test_review_episode_is_the_unit():
    g = torch.Generator().manual_seed(0)
    ep_outcome = (torch.rand(40, generator=g) < 0.8).float()
    lengths = torch.randint(100, 400, (40,), generator=g)
    frame_success = ep_outcome.repeat_interleave(lengths)
    episode_id = torch.arange(40).repeat_interleave(lengths)
    lo, hi = success_ci(frame_success, episode_id)
    assert hi - lo > 0.15, f"CI width {hi - lo:.3f}: with 40 episodes it cannot be this narrow"
