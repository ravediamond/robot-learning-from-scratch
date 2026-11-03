import torch
import torch.nn as nn

from review_w2d5 import evaluate_policy


class _Pixelwise(nn.Module):
    """A 'policy' that reads one exact pixel: any shift changes its output."""

    def forward(self, img, state):
        return img[:, 0, 5, 5][:, None, None].expand(-1, 2, 1)


def test_review_eval_is_deterministic():
    torch.manual_seed(0)
    img = torch.rand(16, 3, 12, 12)
    data = {"img": img, "state": torch.zeros(16, 1), "actions": img[:, 0, 5, 5][:, None, None].expand(-1, 2, 1).clone(),
            "mask": torch.ones(16, 2, dtype=torch.bool)}
    a, b = evaluate_policy(_Pixelwise(), data), evaluate_policy(_Pixelwise(), data)
    assert a == b == 0.0, f"evaluation must see the true images: got {a:.4f}, {b:.4f}"
