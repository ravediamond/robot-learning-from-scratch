"""W11 D2: MuJoCo basics. numpy, not torch."""
from pathlib import Path

import mujoco
import numpy as np

XML = Path(__file__).with_name("so101_like.xml")


def load() -> mujoco.MjModel:
    """Provided."""
    return mujoco.MjModel.from_xml_path(str(XML))


def set_q(model: mujoco.MjModel, data: mujoco.MjData, q: np.ndarray) -> mujoco.MjData:
    """Provided: write joint angles, zero velocity, recompute positions. Returns data."""
    data.qpos[:] = q
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    return data


def n_substeps(control_dt: float, timestep: float) -> int:
    """Physics steps per control step, rounded to the nearest integer, at least 1."""
    # TODO
    raise NotImplementedError


def pd_torque(q: np.ndarray, qd: np.ndarray, q_des: np.ndarray, kp: float, kd: float,
              bias: np.ndarray | None = None) -> np.ndarray:
    """kp * (q_des - q) - kd * qd, plus `bias` (gravity compensation) when given."""
    # TODO
    raise NotImplementedError


def control_step(model: mujoco.MjModel, data: mujoco.MjData, q_des: np.ndarray, kp: float, kd: float,
                 control_dt: float, gravity_comp: bool = False) -> None:
    """One control period: n_substeps physics steps. In EVERY substep recompute the PD torque from the
    current qpos/qvel (bias = data.qfrc_bias when gravity_comp), clip it to the actuator ctrlrange
    (model.actuator_ctrlrange[:, 0] / [:, 1]), write data.ctrl, then mujoco.mj_step."""
    # TODO
    raise NotImplementedError


def ee_pos(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """World position (3,) of the site named "ee" (a COPY). Assumes mj_forward was called."""
    # TODO
    raise NotImplementedError


def ee_jacobian(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Position Jacobian (3, nv) of the "ee" site, with mujoco.mj_jacSite. Assumes mj_forward was called."""
    # TODO
    raise NotImplementedError


def ik_step(model: mujoco.MjModel, data: mujoco.MjData, target: np.ndarray, damping: float = 0.05,
            max_step: float = 0.2) -> np.ndarray:
    """One damped-least-squares step from the pose in `data`. dq = J^T (J J^T + damping^2 I)^-1 (target - ee).
    Scale dq down so that its largest absolute entry is at most max_step, add it to qpos, clip to the joint
    ranges (model.jnt_range), and RETURN the new q (do not modify data)."""
    # TODO
    raise NotImplementedError


def solve_ik(model: mujoco.MjModel, target: np.ndarray, q0: np.ndarray, iters: int = 200, tol: float = 1e-3):
    """Iterate ik_step from q0 on a fresh MjData until the position error is below tol or iters is reached.
    Return (q, final error in metres)."""
    # TODO
    raise NotImplementedError
