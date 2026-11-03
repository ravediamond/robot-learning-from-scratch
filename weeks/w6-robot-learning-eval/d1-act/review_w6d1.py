"""REVIEW. SOLUTION.

BUG: c[:, 0] takes the FIRST action of every stored chunk, i.e. the action each chunk planned for the
time it was made (up to H-1 steps ago): the output averages stale actions and lags ~H/2 steps.
The action for the current time t from a chunk made at s is c[:, t - s].
"""
import torch


class Ensembler:
    def __init__(self, H, m=0.01):
        self.H, self.m, self.t, self.buf = H, m, 0, []

    def step(self, chunk):
        self.buf.append((self.t, chunk))
        self.buf = [(s, c) for s, c in self.buf if self.t - s < self.H]
        acts = torch.stack([c[:, self.t - s] for s, c in self.buf])
        w = torch.exp(-self.m * torch.arange(len(acts), dtype=torch.float32))
        w = w / w.sum()
        self.t += 1
        return (w[:, None, None] * acts).sum(0)
