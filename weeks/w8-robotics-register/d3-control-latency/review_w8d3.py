"""REVIEW. SOLUTION.

BUG: the derivative multiplies by dt instead of dividing: (e - e_prev) * dt is dt^2 times too small, so D
does almost nothing, and its effect changes with the control rate. de/dt = (e - e_prev) / dt.
"""


class PID:
    def __init__(self, kp, ki, kd, dt):
        self.kp, self.ki, self.kd, self.dt = kp, ki, kd, dt
        self.integral, self.prev = 0.0, None

    def step(self, e):
        self.integral += e * self.dt
        d = 0.0 if self.prev is None else (e - self.prev) / self.dt
        self.prev = e
        return self.kp * e + self.ki * self.integral + self.kd * d
