"""REVIEW. SOLUTION.

BUG: (1/30) / 0.002 = 16.67 and int() truncates it to 16. Each control step simulates 32 ms instead of 33.3,
so the simulated arm runs 4% ahead of the log. Use round() (and better: a timestep that divides the
control period exactly, checked by an assert).
"""
import mujoco
import numpy as np


def replay(model: mujoco.MjModel, q_des_log: np.ndarray, kp: float = 20.0, kd: float = 1.0,
           control_dt: float = 1 / 30) -> np.ndarray:
    data = mujoco.MjData(model)
    steps = max(1, round(control_dt / model.opt.timestep))
    out = []
    for q_des in q_des_log:
        for _ in range(steps):
            data.ctrl[:] = np.clip(kp * (q_des - data.qpos) - kd * data.qvel, -3, 3)
            mujoco.mj_step(model, data)
        out.append(data.qpos.copy())
    return np.array(out)
