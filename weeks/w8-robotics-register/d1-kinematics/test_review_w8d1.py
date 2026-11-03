import math

import torch

from review_w8d1 import rot_y


def test_review_right_hand_rule():
    R = rot_y(torch.tensor(math.pi / 2, dtype=torch.float64))
    assert torch.allclose(R @ torch.tensor([1.0, 0, 0], dtype=torch.float64), torch.tensor([0, 0, -1.0], dtype=torch.float64), atol=1e-12), \
        "+90 deg about y must send x to -z (right-hand rule)"
