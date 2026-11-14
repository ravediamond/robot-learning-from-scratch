"""REVIEW. Export step of the deployment pipeline. The single-arm robots work. The new bimanual station runs both
arms' observations in one batch and the runtime refuses the input. Find the bug, fix, replace:
BUG:
"""
import torch


def export(model: torch.nn.Module, obs_dim: int, path: str) -> None:
    model.eval()
    torch.onnx.export(model, (torch.zeros(1, obs_dim),), path, input_names=["obs"], output_names=["act"], dynamo=False)
