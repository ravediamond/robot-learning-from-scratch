import torch

from review_w4d2 import noise_chunks


def test_review_noising_direction():
    abar = torch.cumprod(1 - torch.linspace(1e-4, 0.02, 1000), 0)
    x0, noise = torch.ones(2, 4, 3), torch.zeros(2, 4, 3)
    early = noise_chunks(x0, torch.tensor([0, 0]), noise, abar)
    late = noise_chunks(x0, torch.tensor([999, 999]), noise, abar)
    assert torch.allclose(early, x0, atol=0.01), "at t=0 x_t must be almost the clean chunk"
    assert late.abs().max() < 0.01, "at the last step the signal must be gone"
