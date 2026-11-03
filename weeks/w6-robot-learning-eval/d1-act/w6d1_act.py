"""W6 D1: ACT-style CVAE chunk policy and temporal ensembling. Actions in units of sim2d.MAX_STEP."""
import torch
import torch.nn as nn

import sim2d


class ACTPolicy(nn.Module):
    """encoder: concat(obs (2), flattened chunk (H*2)) -> MLP (256, 256, ReLU) -> mu, logvar (z_dim each)
    decoder: concat(obs, z) -> MLP (256, 256, ReLU) -> chunk (H, 2)"""

    def __init__(self, obs_dim: int = 2, H: int = 8, z_dim: int = 4, hidden: int = 256):
        super().__init__()
        # TODO

    def encode(self, obs, chunk):
        """-> (mu, logvar), each (N, z_dim)."""
        # TODO
        raise NotImplementedError

    def decode(self, obs, z):
        """-> (N, H, 2) in training units."""
        # TODO
        raise NotImplementedError

    def forward(self, obs, chunk, eps):
        """-> (pred chunk, mu, logvar)."""
        # TODO
        raise NotImplementedError

    @torch.no_grad()
    def act(self, obs, generator=None, sample_z: bool = False):
        """Raw actions (N, H, 2): z = 0, or z ~ N(0, I) if sample_z. Multiply by MAX_STEP."""
        # TODO
        raise NotImplementedError


def act_loss(pred, target, mask, mu, logvar, beta: float) -> torch.Tensor:
    """masked L1 (mean over real (step, dim) entries) + beta * mean over the batch of KL to N(0, I)."""
    # TODO
    raise NotImplementedError


def train_act(demos: dict, H: int, steps: int = 3000, batch: int = 256, lr: float = 1e-3, beta: float = 0.01,
              seed: int = 0) -> ACTPolicy:
    # TODO
    raise NotImplementedError


class TemporalEnsembler:
    """step(chunk (N, H, D)) is called once per control step with the chunk predicted at that step.
    Returns the action (N, D) for the current step: weighted average of every stored chunk's prediction
    for the current step, weights exp(-m * i) normalised, i = 0 for the OLDEST chunk."""

    def __init__(self, H: int, m: float = 0.1):
        # TODO
        raise NotImplementedError

    def step(self, chunk: torch.Tensor) -> torch.Tensor:
        # TODO
        raise NotImplementedError
