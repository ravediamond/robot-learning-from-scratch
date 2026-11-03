"""W8 D1: kinematics. Use float64 everywhere."""
import torch

L0, L1, L2, L3 = 0.10, 0.116, 0.135, 0.10


def rot_x(t: torch.Tensor) -> torch.Tensor:
    """3x3 rotation about x by angle t (0-d tensor)."""
    # TODO
    raise NotImplementedError


def rot_y(t: torch.Tensor) -> torch.Tensor:
    # TODO
    raise NotImplementedError


def rot_z(t: torch.Tensor) -> torch.Tensor:
    # TODO
    raise NotImplementedError


def transform(R: torch.Tensor | None = None, p: tuple | torch.Tensor | None = None) -> torch.Tensor:
    """4x4 homogeneous transform (float64) from a rotation and/or a translation (defaults: identity, zero)."""
    # TODO
    raise NotImplementedError


def fk(q: torch.Tensor) -> torch.Tensor:
    """q (5,) -> 4x4 pose of the gripper tip in the base frame (README chain). Differentiable in q."""
    # TODO
    raise NotImplementedError


def jacobian(q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """(3, 5) d tip_position / d q by central differences."""
    # TODO
    raise NotImplementedError


def ik_dls(target: torch.Tensor, q0: torch.Tensor, iters: int = 200, lam: float = 0.01) -> tuple[torch.Tensor, float]:
    """Position-only damped least squares from q0. Returns (q, final position error norm)."""
    # TODO
    raise NotImplementedError
