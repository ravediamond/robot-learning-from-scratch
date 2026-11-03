import math

import torch

import w8d2_camera as m

D = torch.float64


def _pose(seed):
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(3, 3, generator=g, dtype=D))
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    T = torch.eye(4, dtype=D)
    T[:3, :3], T[:3, 3] = Q, torch.randn(3, generator=g, dtype=D) * 0.2
    return T


def test_project_simple():
    K = m.intrinsics(600, 600, 320, 240)
    uv, Z = m.project(torch.tensor([[0.1, -0.05, 0.5]], dtype=D), K, torch.eye(4, dtype=D))
    assert torch.allclose(uv, torch.tensor([[440.0, 180.0]], dtype=D)) and torch.allclose(Z, torch.tensor([0.5], dtype=D))


def test_roundtrip():
    K = m.intrinsics(610, 605, 318, 242)
    T_cw = _pose(0)
    g = torch.Generator().manual_seed(1)
    p_cam = torch.rand(50, 3, generator=g, dtype=D) * torch.tensor([0.4, 0.3, 0.5], dtype=D) + torch.tensor([-0.2, -0.15, 0.4], dtype=D)
    p_world = (p_cam - T_cw[:3, 3]) @ T_cw[:3, :3]                 # inverse transform
    uv, Z = m.project(p_world, K, T_cw)
    assert torch.allclose(Z, p_cam[:, 2], atol=1e-10)
    back = m.backproject(uv, Z, K, torch.linalg.inv(T_cw))
    assert torch.allclose(back, p_world, atol=1e-10)


def test_kabsch_recovers_and_no_reflection():
    T = _pose(3)
    g = torch.Generator().manual_seed(4)
    A = torch.rand(20, 3, generator=g, dtype=D)
    B = A @ T[:3, :3].T + T[:3, 3] + 0.001 * torch.randn(20, 3, generator=g, dtype=D)
    R, t = m.kabsch(A, B)
    assert torch.allclose(R, T[:3, :3], atol=5e-3) and torch.allclose(t, T[:3, 3], atol=5e-3)
    assert m.rigid_rmse(A, B, R, t) < 0.003
    mirrored = A * torch.tensor([1.0, 1.0, -1.0], dtype=D)
    R2, _ = m.kabsch(A, mirrored)
    assert abs(torch.det(R2).item() - 1.0) < 1e-9, "must return a rotation, never a reflection"
