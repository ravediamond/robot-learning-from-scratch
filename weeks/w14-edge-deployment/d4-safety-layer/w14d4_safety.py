"""W14 D4: a deterministic safety filter between the policy and the motors."""
import numpy as np


def clamp_limits(cmd: np.ndarray, limits: np.ndarray, margin: float) -> np.ndarray:
    """Clamp each joint into [low + margin, high - margin]. limits: (J, 2)."""
    return np.clip(cmd, limits[:, 0] + margin, limits[:, 1] - margin)


def rate_limit(prev: np.ndarray, cmd: np.ndarray, max_step: float) -> np.ndarray:
    """Move from prev towards cmd by at most max_step per joint."""
    return prev + np.clip(cmd - prev, -max_step, max_step)


def is_stale(capture_time: float, now: float, max_age: float) -> bool:
    """True if the observation was CAPTURED more than max_age seconds before now."""
    return now - capture_time > max_age


class SafetyFilter:
    def __init__(self, limits: np.ndarray, max_step: float, max_age: float, margin: float = 0.02,
                 initial: np.ndarray | None = None):
        self.limits, self.max_step, self.max_age, self.margin = limits, max_step, max_age, margin
        self.last = initial.copy() if initial is not None else limits.mean(axis=1)
        self.latched = False
        self.events: list[str] = []

    def estop(self) -> None:
        """Latch the stop."""
        self.latched = True
        self.events.append("estop")

    def reset(self) -> None:
        """Explicit operator reset: clears the latch."""
        self.latched = False
        self.events.append("reset")

    def step(self, cmd: np.ndarray, capture_time: float, now: float) -> np.ndarray:
        """Return the command to send and update self.last. In this order:
        - latched: hold self.last, event "latched";
        - any non-finite value in cmd: hold, event "nonfinite";
        - stale observation: hold, event "stale";
        - otherwise: clamp_limits, then rate_limit from self.last (no event).
        Events are appended to self.events. Always return a copy."""
        if self.latched:
            self.events.append("latched")
            return self.last.copy()
        if not np.all(np.isfinite(cmd)):
            self.events.append("nonfinite")
            return self.last.copy()
        if is_stale(capture_time, now, self.max_age):
            self.events.append("stale")
            return self.last.copy()
        safe = rate_limit(self.last, clamp_limits(cmd, self.limits, self.margin), self.max_step)
        self.last = safe
        return safe.copy()
