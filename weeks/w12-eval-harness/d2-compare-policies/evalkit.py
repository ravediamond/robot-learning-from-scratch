"""Provided: shared constants and helpers for week 12."""
import math

OUTCOMES = ("success", "failure", "aborted")
TAXONOMY = ("perception", "grasp", "transport", "placement", "collision", "timeout", "safety_stop")
STAGES = ("reach", "grasp", "lift", "place")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes out of n (you wrote it in W6 D2)."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)
