"""W8 D2: SOLUTION."""
import torch

D = torch.float64


def intrinsics(fx, fy, cx, cy):
    return torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], dtype=D)


def project(p_world, K, T_cam_world):
    p = p_world.to(D) @ T_cam_world[:3, :3].T + T_cam_world[:3, 3]        # (N, 3) camera frame
    Z = p[:, 2]
    uv = (p @ K.T)[:, :2] / Z[:, None]
    return uv, Z


def backproject(uv, Z, K, T_world_cam):
    x = (uv[:, 0] - K[0, 2]) * Z / K[0, 0]
    y = (uv[:, 1] - K[1, 2]) * Z / K[1, 1]
    p_cam = torch.stack([x, y, Z], -1)
    return p_cam @ T_world_cam[:3, :3].T + T_world_cam[:3, 3]


def kabsch(A, B):
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, S, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.det(Vt.T @ U.T))
    R = Vt.T @ torch.diag(torch.tensor([1.0, 1.0, d.item()], dtype=A.dtype)) @ U.T
    return R, cb - ca @ R.T


def rigid_rmse(A, B, R, t):
    return ((A @ R.T + t - B) ** 2).sum(-1).mean().sqrt().item()
