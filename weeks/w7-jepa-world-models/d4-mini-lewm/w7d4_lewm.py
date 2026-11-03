"""W7 D4: a mini LeWM. No EMA, no stop-gradient: SIGReg prevents collapse."""
import torch
import torch.nn as nn

import pixel_world as W
from jepa_ref import sigreg
from probes_ref import apply_probe, copy_last_error, mlp_probe, r2, ridge_probe


class Encoder(nn.Module):
    """See README. (B, 1, 32, 32) -> (B, D)."""

    def __init__(self, D: int = 8):
        super().__init__()
        # TODO

    def forward(self, x):
        # TODO
        raise NotImplementedError


class Predictor(nn.Module):
    """z_next = z + MLP(concat(z, a / W.AMAX)), MLP (D+2) -> 128 -> 128 -> D, ReLU."""

    def __init__(self, D: int = 8):
        super().__init__()
        # TODO

    def forward(self, z, a):
        # TODO
        raise NotImplementedError


def lewm_loss(enc, pred, o0, a, o1, lam: float, generator=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (total loss, prediction MSE). SIGReg (64 directions, `generator`) on cat(z0, z1) when lam > 0."""
    # TODO
    raise NotImplementedError


def train_lewm(data: dict, lam: float, steps: int = 1500, batch: int = 128, lr: float = 1e-3, seed: int = 0):
    """data from W.make_episodes. Each step: random (episode, t) pairs -> (o_t, a_t, o_{t+1}). Adam on both
    modules. Returns (encoder, predictor) in eval mode."""
    # TODO
    raise NotImplementedError


@torch.no_grad()
def latent_report(enc, pred, train: dict, val: dict) -> dict:
    """On VAL episodes: z_std (mean over dims), pred_err (mean ||pred(z_t, a_t) - z_{t+1}||^2), copy_err
    (copy_last_error), ratio = pred_err / copy_err. Probes fitted on TRAIN episodes, R² on VAL for the
    position: lin_r2 (ridge) and mlp_r2 (mlp_probe)."""
    # TODO
    raise NotImplementedError
