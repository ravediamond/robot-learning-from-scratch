"""REVIEW. Observation watchdog of the robot runtime. During a USB hiccup the camera driver delivered a burst of
frames that had been buffered for almost a second; the watchdog let all of them through and the policy acted on
the past. Find the bug, fix, replace:
BUG:
"""


def fresh(frame: dict, now: float, max_age: float = 0.2) -> bool:
    """frame: {"capture_ts": camera timestamp (s), "recv_ts": time the message reached this process (s)}."""
    return now - frame["recv_ts"] <= max_age
