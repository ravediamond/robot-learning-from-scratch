"""W14 D1: a deployable policy module, ONNX export, runtime parity."""
import numpy as np
import onnxruntime as ort
import torch
from torch import nn


class DeployablePolicy(nn.Module):
    """normalise state -> core model -> unnormalise action, with the statistics stored as buffers
    (so they are saved in the state dict and baked into the exported graph)."""

    def __init__(self, core: nn.Module, state_mean, state_std, action_mean, action_std, eps: float = 1e-8):
        super().__init__()
        self.core = core
        self.eps = eps
        # TODO
        raise NotImplementedError

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """state (B, D) in robot units -> action (B, A) in robot units."""
        # TODO
        raise NotImplementedError


def export_onnx(model: nn.Module, example: torch.Tensor, path, dynamic_batch: bool = True) -> None:
    """model.eval(), then torch.onnx.export(model, (example,), str(path), input_names=["obs"], output_names=["act"],
    dynamo=False, and if dynamic_batch: dynamic_axes={"obs": {0: "batch"}, "act": {0: "batch"}})."""
    # TODO
    raise NotImplementedError


def run_onnx(path, x: torch.Tensor) -> np.ndarray:
    """Run the exported graph with onnxruntime (CPUExecutionProvider) on x (as float32 numpy)."""
    # TODO
    raise NotImplementedError


def parity(model: nn.Module, path, x: torch.Tensor) -> float:
    """Max absolute difference between the PyTorch model in EVAL mode (no grad) and the ONNX graph on x."""
    # TODO
    raise NotImplementedError
