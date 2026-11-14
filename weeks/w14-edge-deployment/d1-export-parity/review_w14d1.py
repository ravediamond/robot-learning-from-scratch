"""REVIEW. SOLUTION.

BUG: the graph is traced with a batch of 1 and no dynamic axes, so the batch dimension is frozen at 1. Declare
the batch axis dynamic for the input and the output (and, for TensorRT, give min/opt/max shapes when building).
"""
import torch


def export(model: torch.nn.Module, obs_dim: int, path: str) -> None:
    model.eval()
    torch.onnx.export(model, (torch.zeros(1, obs_dim),), path, input_names=["obs"], output_names=["act"], dynamo=False,
                      dynamic_axes={"obs": {0: "batch"}, "act": {0: "batch"}})
