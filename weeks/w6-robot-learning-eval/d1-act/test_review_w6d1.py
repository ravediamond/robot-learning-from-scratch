import torch

from review_w6d1 import Ensembler


def test_review_action_for_the_current_time():
    H = 5
    e = Ensembler(H, m=0.01)
    for k in range(12):
        chunk = (k + torch.arange(H, dtype=torch.float32))[None, :, None]   # chunk made at k predicts "time" k..k+H-1
        a = e.step(chunk)
        assert abs(a.item() - k) < 1e-5, f"at t={k} every stored chunk predicts {k}, got {a.item():.2f}"
