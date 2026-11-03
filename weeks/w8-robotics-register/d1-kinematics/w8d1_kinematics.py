"""W8 D1: SOLUTION."""
import torch

L0, L1, L2, L3 = 0.10, 0.116, 0.135, 0.10
D = torch.float64


def _t(t):
    return torch.as_tensor(t, dtype=D)


def rot_x(t):
    t = _t(t)
    c, s, o, z = torch.cos(t), torch.sin(t), torch.ones_like(t), torch.zeros_like(t)
    return torch.stack([torch.stack([o, z, z]), torch.stack([z, c, -s]), torch.stack([z, s, c])])


def rot_y(t):
    t = _t(t)
    c, s, o, z = torch.cos(t), torch.sin(t), torch.ones_like(t), torch.zeros_like(t)
    return torch.stack([torch.stack([c, z, s]), torch.stack([z, o, z]), torch.stack([-s, z, c])])


def rot_z(t):
    t = _t(t)
    c, s, o, z = torch.cos(t), torch.sin(t), torch.ones_like(t), torch.zeros_like(t)
    return torch.stack([torch.stack([c, -s, z]), torch.stack([s, c, z]), torch.stack([z, z, o])])


def transform(R=None, p=None):
    top = torch.eye(3, dtype=D) if R is None else R.to(D)
    pv = torch.zeros(3, dtype=D) if p is None else torch.as_tensor(p, dtype=D)
    return torch.cat([torch.cat([top, pv[:, None]], 1), torch.tensor([[0, 0, 0, 1.0]], dtype=D)], 0)


def fk(q):
    T = transform(rot_z(q[0])) @ transform(p=(0, 0, L0))
    T = T @ transform(rot_y(q[1])) @ transform(p=(L1, 0, 0))
    T = T @ transform(rot_y(q[2])) @ transform(p=(L2, 0, 0))
    T = T @ transform(rot_y(q[3])) @ transform(rot_x(q[4])) @ transform(p=(L3, 0, 0))
    return T


def jacobian(q, eps=1e-6):
    q = q.to(D)
    cols = []
    for i in range(len(q)):
        dq = torch.zeros_like(q)
        dq[i] = eps
        cols.append((fk(q + dq)[:3, 3] - fk(q - dq)[:3, 3]) / (2 * eps))
    return torch.stack(cols, 1)


def ik_dls(target, q0, iters=200, lam=0.01):
    q = q0.clone().to(D)
    for _ in range(iters):
        e = target.to(D) - fk(q)[:3, 3]
        if e.norm() < 1e-6:
            break
        J = jacobian(q)
        q = q + J.T @ torch.linalg.solve(J @ J.T + lam ** 2 * torch.eye(3, dtype=D), e)
    return q, (target.to(D) - fk(q)[:3, 3]).norm().item()
