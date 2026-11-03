import torch

from review_w8d2 import project_cam


def test_review_perspective_division():
    p = torch.tensor([[0.0, 0.0, 1.0], [0.3, -0.2, 0.6]], dtype=torch.float64)
    uv = project_cam(p, 600, 600, 320, 240)
    assert torch.allclose(uv, torch.tensor([[320.0, 240.0], [620.0, 40.0]], dtype=torch.float64)), uv.tolist()
