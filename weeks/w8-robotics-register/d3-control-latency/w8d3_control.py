"""W8 D3: PID, delay, and chunk switching."""
import math

import torch


class PID:
    def __init__(self, kp: float, ki: float, kd: float, dt: float):
        # TODO
        raise NotImplementedError

    def step(self, error: float) -> float:
        """u = kp e + ki * integral(e dt) + kd * de/dt. First call: derivative 0."""
        # TODO
        raise NotImplementedError


def simulate_plant(pid: PID, steps: int, delay: int, disturbance: float, target: float = 1.0) -> list[float]:
    """Velocity-controlled plant: x_{t+1} = x_t + dt * (u_{t-delay} + disturbance), x_0 = 0, dt = pid.dt.
    u_{t-delay} = 0 for t < delay. The PID sees error = target - x_t each step. Returns [x_0 .. x_steps]."""
    # TODO
    raise NotImplementedError


def target_traj(T: int) -> torch.Tensor:
    return 0.5 * torch.sin(2 * math.pi * torch.arange(T, dtype=torch.float64) / 60)


def plan_chunk(t_obs: int, H: int, T: int, offset: float) -> torch.Tensor:
    """PROVIDED policy: commands for times t_obs+1 .. t_obs+H = the target there + a per-chunk offset
    (a different 'sample' each call)."""
    idx = torch.arange(t_obs + 1, t_obs + 1 + H).clamp(max=T - 1)
    return target_traj(T)[idx] + offset


def run_chunked(mode: str, delay: int = 6, H: int = 20, exec_h: int = 10, T: int = 240, blend_m: int = 5,
                seed: int = 0) -> dict:
    """Position-controlled robot (it goes exactly where commanded). A new chunk is REQUESTED every exec_h
    steps (at t = 0, exec_h, 2 exec_h, ...) with plan_chunk(t, H, T, offset), offset ~ 0.05 * N(0, 1) from
    Generator(seed). It ARRIVES at t + delay; before the first arrival the command is target(0).
    From the arrival until the next arrival, the command at time s is taken from the newest chunk c made at t:
      naive: c[s - (t + delay)]          shift: c[s - t - 1]          (indices clamped to H - 1)
      blend: like shift, but for the first blend_m steps after the arrival, mix with the previous chunk's
             command for the same time s (also shift-indexed): w = (k + 1) / (blend_m + 1), k = 0..blend_m-1,
             command = (1 - w) * old + w * new.
    Returns {"commands": (T,), "tracking_err": mean |command - target| over t >= 20,
             "max_jump": max |command_s - command_{s-1}| over s >= 20}."""
    # TODO
    raise NotImplementedError
