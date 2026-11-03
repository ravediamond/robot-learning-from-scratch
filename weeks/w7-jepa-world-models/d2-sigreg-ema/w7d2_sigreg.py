"""W7 D2: SIGReg, EMA, stop-gradient."""
import torch
import torch.nn as nn


def epps_pulley(p: torch.Tensor, T: float = 3.0, K: int = 17) -> torch.Tensor:
    """p (N, M): M one-dimensional samples of size N. Returns (M,) Epps-Pulley statistics vs N(0, 1)
    (formula in the README, multiplied by N, trapezoid rule over K points in [-T, T])."""
    # TODO
    raise NotImplementedError


def sigreg(z: torch.Tensor, n_dirs: int = 64, generator: torch.Generator | None = None) -> torch.Tensor:
    """z (N, d). Random unit directions U (d, n_dirs) from `generator`, project, mean Epps-Pulley."""
    # TODO
    raise NotImplementedError


@torch.no_grad()
def ema_update(target: nn.Module, online: nn.Module, m: float) -> None:
    """In place: every target param <- m * target + (1 - m) * online."""
    # TODO
    raise NotImplementedError


def jepa_ema_loss(online: nn.Module, target: nn.Module, predictor: nn.Module, x_ctx: torch.Tensor,
                  x_tgt: torch.Tensor) -> torch.Tensor:
    """MSE between predictor(online(x_ctx)) and target(x_tgt), with NO gradient into the target encoder."""
    # TODO
    raise NotImplementedError
