"""L10 (30 min). Quaternions (w, x, y, z), batched over leading dimensions."""
import torch


def quat_mul(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Hamilton product p * q, shapes (..., 4) broadcastable."""
    pw, px, py, pz = p.unbind(-1)
    qw, qx, qy, qz = q.unbind(-1)
    return torch.stack([pw * qw - px * qx - py * qy - pz * qz,
                        pw * qx + px * qw + py * qz - pz * qy,
                        pw * qy - px * qz + py * qw + pz * qx,
                        pw * qz + px * qy - py * qx + pz * qw], dim=-1)


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=q.dtype)


def rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v (..., 3) by unit quaternions q (..., 4): the vector part of q * (0, v) * conj(q)."""
    vq = torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)
    return quat_mul(quat_mul(q, vq), quat_conj(q))[..., 1:]


def from_axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """axis (..., 3) (not necessarily unit), angle (...) in radians -> unit quaternion (..., 4)."""
    axis = axis / axis.norm(dim=-1, keepdim=True)
    half = angle[..., None] / 2
    return torch.cat([half.cos(), axis * half.sin()], dim=-1)
