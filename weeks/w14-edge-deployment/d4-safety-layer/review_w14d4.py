"""REVIEW. SOLUTION.

BUG: freshness was measured from the time the frame was RECEIVED. Buffered frames arrive late but look fresh on
arrival. What matters is how old the world in the image is: measure from the capture timestamp (on a clock
synchronised with this process, or from the camera's hardware timestamp mapped to it).
"""


def fresh(frame: dict, now: float, max_age: float = 0.2) -> bool:
    return now - frame["capture_ts"] <= max_age
