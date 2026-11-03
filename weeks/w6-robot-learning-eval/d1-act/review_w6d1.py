"""REVIEW. Temporal ensembling for an ACT policy at 30 Hz. The arm moves smoothly, but it lags
behind the motion it was shown, more so for longer chunks. Find the bug, fix, replace:
BUG:
"""
import torch


class Ensembler:
    def __init__(self, H: int, m: float = 0.01):
        self.H, self.m, self.t, self.buf = H, m, 0, []

    def step(self, chunk: torch.Tensor) -> torch.Tensor:
        """chunk (N, H, D) predicted now -> action (N, D) for now."""
        self.buf.append((self.t, chunk))
        self.buf = [(s, c) for s, c in self.buf if self.t - s < self.H]
        acts = torch.stack([c[:, 0] for s, c in self.buf])
        w = torch.exp(-self.m * torch.arange(len(acts), dtype=torch.float32))
        w = w / w.sum()
        self.t += 1
        return (w[:, None, None] * acts).sum(0)
