"""W8 D2: pinhole camera and rigid fit. float64."""
import torch


def intrinsics(fx: float, fy: float, cx: float, cy: float) -> torch.Tensor:
    # TODO
    raise NotImplementedError


def project(p_world: torch.Tensor, K: torch.Tensor, T_cam_world: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(N, 3) world points -> (pixels (N, 2), depth Z (N,)) in the camera."""
    # TODO
    raise NotImplementedError


def backproject(uv: torch.Tensor, Z: torch.Tensor, K: torch.Tensor, T_world_cam: torch.Tensor) -> torch.Tensor:
    """(N, 2) pixels + (N,) depths -> (N, 3) world points."""
    # TODO
    raise NotImplementedError


def kabsch(A: torch.Tensor, B: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Least-squares R (3x3, det +1), t (3,) with B ≈ A @ R.T + t. A, B (N, 3)."""
    # TODO
    raise NotImplementedError


def rigid_rmse(A, B, R, t) -> float:
    """Root mean square distance between A @ R.T + t and B."""
    # TODO
    raise NotImplementedError
