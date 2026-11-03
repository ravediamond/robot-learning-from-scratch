import torch

import sim2d
from review_w4d5 import execute_step


def test_review_units_and_clip():
    pos = torch.zeros(2, 2)
    chunk = torch.tensor([[[1.0, 0.0]], [[0.0, -0.5]]])          # policy output in units of MAX_STEP
    new = execute_step(pos, chunk, 0)
    assert torch.allclose(new, torch.tensor([[sim2d.MAX_STEP, 0.0], [0.0, -0.5 * sim2d.MAX_STEP]]))
    huge = execute_step(torch.zeros(1, 2), torch.tensor([[[30.0, 40.0]]]), 0)
    assert huge.norm() <= sim2d.MAX_STEP + 1e-6, "every command must be clipped before it reaches the robot"
