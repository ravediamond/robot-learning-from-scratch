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
        self.register_buffer("state_mean", torch.as_tensor(state_mean, dtype=torch.float32))
        self.register_buffer("state_std", torch.as_tensor(state_std, dtype=torch.float32))
        self.register_buffer("action_mean", torch.as_tensor(action_mean, dtype=torch.float32))
        self.register_buffer("action_std", torch.as_tensor(action_std, dtype=torch.float32))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """state (B, D) in robot units -> action (B, A) in robot units."""
        z = (state - self.state_mean) / (self.state_std + self.eps)
        return self.core(z) * self.action_std + self.action_mean


def export_onnx(model: nn.Module, example: torch.Tensor, path, dynamic_batch: bool = True) -> None:
    """model.eval(), then torch.onnx.export(model, (example,), str(path), input_names=["obs"], output_names=["act"],
    dynamo=False, and if dynamic_batch: dynamic_axes={"obs": {0: "batch"}, "act": {0: "batch"}})."""
    model.eval()
    kwargs = {"input_names": ["obs"], "output_names": ["act"], "dynamo": False}
    if dynamic_batch:
        kwargs["dynamic_axes"] = {"obs": {0: "batch"}, "act": {0: "batch"}}
    torch.onnx.export(model, (example,), str(path), **kwargs)


def run_onnx(path, x: torch.Tensor) -> np.ndarray:
    """Run the exported graph with onnxruntime (CPUExecutionProvider) on x (as float32 numpy)."""
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return sess.run(None, {"obs": x.detach().cpu().numpy().astype(np.float32)})[0]


def parity(model: nn.Module, path, x: torch.Tensor) -> float:
    """Max absolute difference between the PyTorch model in EVAL mode (no grad) and the ONNX graph on x."""
    model.eval()
    with torch.no_grad():
        ref = model(x).cpu().numpy()
    return float(np.abs(ref - run_onnx(path, x)).max())
