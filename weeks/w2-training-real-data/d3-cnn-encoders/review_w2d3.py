"""REVIEW. Frames from the video decoder arrive as uint8 (B, H, W, 3). This converts them for
the CNN. Shapes are right, values are in [0, 1], the policy trains, but it is clearly worse
than the same policy trained from LeRobot's own loader. Find the bug, fix, replace:
BUG:
"""
import torch


def to_model_input(frames: torch.Tensor) -> torch.Tensor:
    """(B, H, W, 3) uint8 -> (B, 3, H, W) float32 in [0, 1]."""
    B, H, W, C = frames.shape
    return frames.reshape(B, C, H, W).float() / 255.0
