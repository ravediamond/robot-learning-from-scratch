"""W13 D1: dataset QA checks and a quarantine report."""
import hashlib

import numpy as np


def longest_frozen_run(frame_hash: np.ndarray) -> int:
    """Length of the longest run of consecutive identical values (1 if none repeat, 0 if empty)."""
    # TODO
    raise NotImplementedError


def dead_joints(state: np.ndarray, min_range: float = 1e-3) -> list[int]:
    """Joints whose range (max - min over the episode) is below min_range, while at least one other joint
    moves more than min_range (an arm that does not move at all is a different problem)."""
    # TODO
    raise NotImplementedError


def limit_fraction(state: np.ndarray, limits: np.ndarray, margin: float = 0.02) -> float:
    """Fraction of frames where ANY joint is within `margin` of its lower or upper limit, or beyond it.
    limits: (J, 2) array of [low, high]."""
    # TODO
    raise NotImplementedError


def robust_z(x: np.ndarray) -> np.ndarray:
    """0.6745 * (x - median) / MAD, MAD = median(|x - median|). If MAD == 0, return zeros."""
    # TODO
    raise NotImplementedError


def content_hash(ep: dict) -> str:
    """SHA-256 of the episode's state and action arrays (as float64, C order, bytes) and their shapes.
    Metadata (id, station, ...) is NOT part of it: two uploads of the same recording must collide."""
    # TODO
    raise NotImplementedError


def qa_report(eps: list[dict], limits: np.ndarray, fps: float, max_frozen_s: float = 2.0,
              max_limit_frac: float = 0.1, z_thresh: float = 3.5) -> dict:
    """Run every check and return {"quarantine": {episode id: sorted list of reasons}, "duplicates": sorted list of
    (kept id, duplicate id)}. Reasons (strings): "frozen_camera" (longest frozen run > max_frozen_s * fps frames),
    "dead_joint", "limits" (limit_fraction > max_limit_frac), "length_outlier" (|robust_z of lengths| > z_thresh),
    "duplicate" (same content hash as an EARLIER episode in the list; the first one is kept and not flagged).
    Only episodes with at least one reason appear in "quarantine"."""
    # TODO
    raise NotImplementedError
