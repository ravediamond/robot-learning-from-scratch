"""Run AFTER writing your guesses AND after your tests pass."""
import math

import torch

import w8d1_kinematics as m

q0 = torch.zeros(5, dtype=torch.float64)
print("1", m.fk(q0)[:3, 3].tolist(), m.fk(torch.tensor([0, -math.pi / 2, 0, 0, 0], dtype=torch.float64))[:3, 3].tolist())
print("2 reach", 0.116 + 0.135 + 0.10)
print("3 rank at q=0:", torch.linalg.matrix_rank(m.jacobian(q0)).item(), " (stretched: no joint moves the tip along the arm)")
print("4 roll column:", m.jacobian(q0)[:, 4].tolist())
q, res = m.ik_dls(torch.tensor([0.5, 0.0, 0.1], dtype=torch.float64), torch.full((5,), 0.1, dtype=torch.float64))
print("5 out of reach: residual", round(res, 4), " tip", m.fk(q)[:3, 3].tolist())
