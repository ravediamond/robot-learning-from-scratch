"""W2 D3: CNN encoders."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv2d_unfold(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None,
                  stride: int = 1, padding: int = 0) -> torch.Tensor:
    """Same result as F.conv2d(x, weight, bias, stride, padding), using F.unfold + matmul.
    x: (B, C, H, W), weight: (C_out, C, k, k)."""
    # TODO
    raise NotImplementedError


def receptive_field(layers: list[tuple[int, int]]) -> int:
    """layers: [(kernel, stride), ...] in order. Receptive field (pixels) of one output unit."""
    # TODO
    raise NotImplementedError


class SpatialSoftmax(nn.Module):
    """(B, C, H, W) -> (B, 2C): per channel, expected (x, y) under softmax over H*W positions.
    Coordinates in [-1, 1] (x along W, y along H). Output order: [x_0, y_0, x_1, y_1, ...]."""

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # TODO
        raise NotImplementedError


class CNNEncoder(nn.Module):
    """Image (B, 3, H, W) in [0, 1] -> (B, out_dim).

    conv(3->32, k5, s2) GN ReLU, conv(32->32, k3, s1) GN ReLU, conv(32->32, k3, s1) GN ReLU,
    then SpatialSoftmax -> Linear(64, out_dim). GroupNorm with 8 groups. No BatchNorm.
    """

    def __init__(self, out_dim: int = 64):
        super().__init__()
        # TODO

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # TODO
        raise NotImplementedError


def random_shift(imgs: torch.Tensor, pad: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """DrQ augmentation. imgs: (B, C, H, W). Replicate-pad by `pad`, then crop an H x W window
    at an independent random integer offset in [0, 2*pad] for each sample. pad=0 -> identity."""
    # TODO
    raise NotImplementedError
