"""REVIEW. SOLUTION.

BUG: divides by the Euclidean distance to the point instead of its depth Z along the optical axis. At the
image centre the two are equal; off-axis the distance is larger than Z, so points are pulled inward.
"""
import torch


def project_cam(p_cam, fx, fy, cx, cy):
    Z = p_cam[:, 2]
    return torch.stack([fx * p_cam[:, 0] / Z + cx, fy * p_cam[:, 1] / Z + cy], -1)
