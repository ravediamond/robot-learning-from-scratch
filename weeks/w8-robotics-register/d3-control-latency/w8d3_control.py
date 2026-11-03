"""W8 D3: SOLUTION."""
import math

import torch


class PID:
    def __init__(self, kp, ki, kd, dt):
        self.kp, self.ki, self.kd, self.dt = kp, ki, kd, dt
        self.integral, self.prev = 0.0, None

    def step(self, error):
        self.integral += error * self.dt
        d = 0.0 if self.prev is None else (error - self.prev) / self.dt
        self.prev = error
        return self.kp * error + self.ki * self.integral + self.kd * d


def simulate_plant(pid, steps, delay, disturbance, target=1.0):
    x, xs, us = 0.0, [0.0], []
    for t in range(steps):
        us.append(pid.step(target - x))
        u = us[t - delay] if t >= delay else 0.0
        x = x + pid.dt * (u + disturbance)
        xs.append(x)
    return xs


def target_traj(T):
    return 0.5 * torch.sin(2 * math.pi * torch.arange(T, dtype=torch.float64) / 60)


def plan_chunk(t_obs, H, T, offset):
    idx = torch.arange(t_obs + 1, t_obs + 1 + H).clamp(max=T - 1)
    return target_traj(T)[idx] + offset


def run_chunked(mode, delay=6, H=20, exec_h=10, T=240, blend_m=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    tgt = target_traj(T)
    chunks = {}                                                   # arrival time -> (t_made, chunk)
    for t in range(0, T, exec_h):
        chunks[t + delay] = (t, plan_chunk(t, H, T, 0.05 * torch.randn(1, generator=g, dtype=torch.float64).item()))
    cmd = torch.zeros(T, dtype=torch.float64)
    cur, prev, since = None, None, 0
    for s in range(T):
        if s in chunks:
            prev, cur, since = cur, chunks[s], 0
        if cur is None:
            cmd[s] = tgt[0]
            continue

        def from_chunk(ch, naive):
            t_made, c = ch
            k = s - (t_made + delay) if naive else s - t_made - 1
            return c[min(max(k, 0), H - 1)]

        new = from_chunk(cur, mode == "naive")
        if mode == "blend" and prev is not None and since < blend_m:
            w = (since + 1) / (blend_m + 1)
            new = (1 - w) * from_chunk(prev, False) + w * new
        cmd[s] = new
        since += 1
    jumps = (cmd[1:] - cmd[:-1]).abs()
    return {"commands": cmd, "tracking_err": (cmd[20:] - tgt[20:]).abs().mean().item(), "max_jump": jumps[19:].max().item()}
