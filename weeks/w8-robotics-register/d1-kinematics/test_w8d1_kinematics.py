import math

import torch

import w8d1_kinematics as m

D = torch.float64


def test_rotations():
    t = torch.tensor(math.pi / 2, dtype=D)
    x = torch.tensor([1.0, 0, 0], dtype=D)
    assert torch.allclose(m.rot_z(t) @ x, torch.tensor([0, 1.0, 0], dtype=D), atol=1e-12)
    assert torch.allclose(m.rot_y(t) @ x, torch.tensor([0, 0, -1.0], dtype=D), atol=1e-12)
    assert torch.allclose(m.rot_x(t) @ torch.tensor([0, 1.0, 0], dtype=D), torch.tensor([0, 0, 1.0], dtype=D), atol=1e-12)
    R = m.rot_y(torch.tensor(0.3, dtype=D)) @ m.rot_x(torch.tensor(-1.1, dtype=D))
    assert torch.allclose(R @ R.T, torch.eye(3, dtype=D), atol=1e-12) and abs(torch.det(R) - 1) < 1e-12


def test_transform_compose():
    T = m.transform(m.rot_z(torch.tensor(math.pi / 2, dtype=D)), (1.0, 0.0, 0.0)) @ m.transform(p=(1.0, 0.0, 0.0))
    assert torch.allclose(T[:3, 3], torch.tensor([1.0, 1.0, 0.0], dtype=D), atol=1e-12)
    assert T.shape == (4, 4) and T.dtype == D and torch.equal(T[3], torch.tensor([0, 0, 0, 1.0], dtype=D))


def test_fk_known_poses():
    assert torch.allclose(m.fk(torch.zeros(5, dtype=D))[:3, 3], torch.tensor([0.351, 0, 0.10], dtype=D), atol=1e-12)
    up = m.fk(torch.tensor([0, -math.pi / 2, 0, 0, 0], dtype=D))[:3, 3]
    assert torch.allclose(up, torch.tensor([0, 0, 0.451], dtype=D), atol=1e-12), "shoulder lift -90 deg points up"
    pan = m.fk(torch.tensor([math.pi / 2, 0, 0, 0, 0], dtype=D))[:3, 3]
    assert torch.allclose(pan, torch.tensor([0, 0.351, 0.10], dtype=D), atol=1e-12)


def test_jacobian_matches_autograd():
    q = torch.tensor([0.3, -0.4, 0.7, 0.2, 0.5], dtype=D)
    J = m.jacobian(q)
    Ja = torch.autograd.functional.jacobian(lambda x: m.fk(x)[:3, 3], q)
    assert J.shape == (3, 5) and torch.allclose(J, Ja, atol=1e-7)
    assert torch.allclose(J[:, 4], torch.zeros(3, dtype=D), atol=1e-9), "wrist roll does not move the tip"


def test_ik():
    q_true = torch.tensor([0.4, -0.6, 0.9, 0.3, 0.0], dtype=D)
    target = m.fk(q_true)[:3, 3]
    q, err = m.ik_dls(target, torch.zeros(5, dtype=D) + 0.1)
    assert err < 1e-3 and torch.allclose(m.fk(q)[:3, 3], target, atol=1e-3)
    q, err = m.ik_dls(torch.tensor([0.6, 0.0, 0.1], dtype=D), torch.full((5,), 0.1, dtype=D))
    assert torch.isfinite(q).all() and 0.24 < err < 0.5, "out of reach: finite q, residual at least the distance beyond reach (0.249)"
