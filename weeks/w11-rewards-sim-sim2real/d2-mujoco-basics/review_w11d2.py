"""REVIEW. Replay of a recorded 30 Hz command log in simulation. The simulated arm finishes the motion about
4% early compared with the real log. Find the bug, fix, replace:
BUG:
"""
import mujoco
import numpy as np


def replay(model: mujoco.MjModel, q_des_log: np.ndarray, kp: float = 20.0, kd: float = 1.0,
           control_dt: float = 1 / 30) -> np.ndarray:
    """q_des_log (T, nu): one target per control step. Returns the joint trajectory (T, nq)."""
    data = mujoco.MjData(model)
    steps = int(control_dt / model.opt.timestep)
    out = []
    for q_des in q_des_log:
        for _ in range(steps):
            data.ctrl[:] = np.clip(kp * (q_des - data.qpos) - kd * data.qvel, -3, 3)
            mujoco.mj_step(model, data)
        out.append(data.qpos.copy())
    return np.array(out)
