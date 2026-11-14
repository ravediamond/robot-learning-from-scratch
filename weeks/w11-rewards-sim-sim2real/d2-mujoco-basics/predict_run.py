"""Run AFTER writing your guesses AND after your tests pass (~5 s)."""
import sys as _sys


def _unfinished_lab_hint(exc_type, exc, tb):
    if exc_type is NotImplementedError:
        print("\nThis predict script runs YOUR lab code, which is not finished yet (a TODO still raises NotImplementedError)."
              "\nMake `uv run pytest . -k \"not review\"` green in this folder first, then run me again.", file=_sys.stderr)
    else:
        _sys.__excepthook__(exc_type, exc, tb)


_sys.excepthook = _unfinished_lab_hint
import mujoco
import numpy as np

import w11d2_mujoco as m

print("1.", int((1 / 30) / 0.002), round((1 / 30) / 0.002), " (16.67 steps: int truncates)")
model = m.load()
q_des = np.array([0.0, 1.2, 0.0, 0.0, 0.0])
for label, kp, comp in (("2. kp=20, no compensation", 20, False), ("3. kp=20, gravity compensation", 20, True),
                        ("4. kp=5,  no compensation", 5, False)):
    data = mujoco.MjData(model)
    for _ in range(150):
        m.control_step(model, data, q_des, kp, 1.0, 0.02, gravity_comp=comp)
    print(f"{label}: error at shoulder_lift {q_des[1] - data.qpos[1]:+.4f} rad")
data = mujoco.MjData(model)
J = m.ee_jacobian(model, data)
print("5. J shape", J.shape, " column norms at the zero pose:", np.round(np.linalg.norm(J, axis=0), 3))
target = m.ee_pos(model, m.set_q(model, mujoco.MjData(model), np.array([0.6, 0.5, -0.8, 0.4, 0.0])))
q = np.zeros(5)
for i in range(300):
    data = m.set_q(model, mujoco.MjData(model), q)
    if np.linalg.norm(target - m.ee_pos(model, data)) < 1e-3:
        break
    q = m.ik_step(model, data, target, damping=0.05)
print(f"6. under 1 mm after {i} iterations")
