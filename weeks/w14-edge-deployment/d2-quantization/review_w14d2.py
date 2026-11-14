"""REVIEW. Quantisation-aware fine-tuning of the gripper head. The loss is flat from the first step and the weights
never change, though the learning rate is fine. Find the bug, fix, replace:
BUG:
"""
import torch


def fake_quant(x: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.clamp(torch.round(x / scale), -127, 127) * scale
