import torch

from review_w2d3 import to_model_input


def test_review_channels_first():
    frames = torch.randint(0, 256, (2, 4, 6, 3), dtype=torch.uint8)
    out = to_model_input(frames)
    assert out.shape == (2, 3, 4, 6) and out.dtype == torch.float32
    assert torch.allclose(out, frames.permute(0, 3, 1, 2).float() / 255), "pixels moved across channels/rows"
