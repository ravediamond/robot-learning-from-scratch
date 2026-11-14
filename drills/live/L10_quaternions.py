"""L10 (30 min). Quaternions (w, x, y, z), batched over leading dimensions."""
import torch


def quat_mul(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Hamilton product p * q, shapes (..., 4) broadcastable."""
    # TODO
    raise NotImplementedError


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    # TODO
    raise NotImplementedError


def rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v (..., 3) by unit quaternions q (..., 4): the vector part of q * (0, v) * conj(q)."""
    # TODO
    raise NotImplementedError


def from_axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """axis (..., 3) (not necessarily unit), angle (...) in radians -> unit quaternion (..., 4)."""
    # TODO
    raise NotImplementedError
