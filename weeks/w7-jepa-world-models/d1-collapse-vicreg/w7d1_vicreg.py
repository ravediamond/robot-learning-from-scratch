"""W7 D1: collapse and VICReg."""
import torch
import torch.nn as nn


def vicreg_loss(z1: torch.Tensor, z2: torch.Tensor, sim_w: float = 25.0, var_w: float = 25.0, cov_w: float = 1.0,
                use_cov: bool = True, eps: float = 1e-4) -> torch.Tensor:
    """z1, z2 (N, d). See README. use_cov=False drops the covariance term."""
    # TODO
    raise NotImplementedError


def effective_rank(z: torch.Tensor) -> float:
    """exp(entropy) of the normalised singular values of the CENTRED z (N, d)."""
    # TODO
    raise NotImplementedError


def train_siamese(data: torch.Tensor, kind: str, steps: int = 500, batch: int = 256, seed: int = 0):
    """Encoder MLP 2 -> 64 -> 64 -> 8 (ReLU). Each step: x = random batch; two views x + 0.05*noise.
    kind: "invariance" (MSE between views only), "vicreg", or "no_cov" (vicreg with use_cov=False).
    Adam 1e-3. Returns (encoder in eval mode, last loss as float)."""
    # TODO
    raise NotImplementedError
