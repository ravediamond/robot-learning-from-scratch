"""REVIEW. Projects a detected object into the top camera image. Close to the image centre it
matches the detector; toward the edges the projected points drift inward by tens of pixels.
Find the bug, fix, replace:
BUG:
"""
import torch


def project_cam(p_cam: torch.Tensor, fx: float, fy: float, cx: float, cy: float) -> torch.Tensor:
    """(N, 3) points in the CAMERA frame -> (N, 2) pixels."""
    dist = p_cam.norm(dim=-1)
    return torch.stack([fx * p_cam[:, 0] / dist + cx, fy * p_cam[:, 1] / dist + cy], -1)
