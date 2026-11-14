"""W14 D4: a deterministic safety filter between the policy and the motors."""
import numpy as np


def clamp_limits(cmd: np.ndarray, limits: np.ndarray, margin: float) -> np.ndarray:
    """Clamp each joint into [low + margin, high - margin]. limits: (J, 2)."""
    # TODO
    raise NotImplementedError


def rate_limit(prev: np.ndarray, cmd: np.ndarray, max_step: float) -> np.ndarray:
    """Move from prev towards cmd by at most max_step per joint."""
    # TODO
    raise NotImplementedError


def is_stale(capture_time: float, now: float, max_age: float) -> bool:
    """True if the observation was CAPTURED more than max_age seconds before now."""
    # TODO
    raise NotImplementedError


class SafetyFilter:
    def __init__(self, limits: np.ndarray, max_step: float, max_age: float, margin: float = 0.02,
                 initial: np.ndarray | None = None):
        self.limits, self.max_step, self.max_age, self.margin = limits, max_step, max_age, margin
        self.last = initial.copy() if initial is not None else limits.mean(axis=1)
        self.latched = False
        self.events: list[str] = []

    def estop(self) -> None:
        """Latch the stop."""
        # TODO
        raise NotImplementedError

    def reset(self) -> None:
        """Explicit operator reset: clears the latch."""
        # TODO
        raise NotImplementedError

    def step(self, cmd: np.ndarray, capture_time: float, now: float) -> np.ndarray:
        """Return the command to send and update self.last. In this order:
        - latched: hold self.last, event "latched";
        - any non-finite value in cmd: hold, event "nonfinite";
        - stale observation: hold, event "stale";
        - otherwise: clamp_limits, then rate_limit from self.last (no event).
        Events are appended to self.events. Always return a copy."""
        # TODO
        raise NotImplementedError
