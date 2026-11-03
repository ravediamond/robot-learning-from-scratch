"""REVIEW. A PID for the gripper position. With kd > 0 the gripper is sluggish instead of damped,
and changing the control rate from 30 Hz to 100 Hz changes the behaviour a lot.
Find the bug, fix, replace:
BUG:
"""


class PID:
    def __init__(self, kp, ki, kd, dt):
        self.kp, self.ki, self.kd, self.dt = kp, ki, kd, dt
        self.integral, self.prev = 0.0, None

    def step(self, e: float) -> float:
        self.integral += e * self.dt
        d = 0.0 if self.prev is None else (e - self.prev) * self.dt
        self.prev = e
        return self.kp * e + self.ki * self.integral + self.kd * d
