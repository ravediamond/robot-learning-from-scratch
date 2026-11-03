"""REVIEW. SOLUTION.

BUG: reshape reinterprets the HWC memory as CHW: pixels get scrambled across channels and rows
(shape right, image wrong). Channels-last to channels-first needs permute(0, 3, 1, 2).
"""
import torch


def to_model_input(frames):
    return frames.permute(0, 3, 1, 2).float().div(255.0).contiguous()
