"""W13 D1: dataset QA checks and a quarantine report."""
import hashlib

import numpy as np


def longest_frozen_run(frame_hash: np.ndarray) -> int:
    """Length of the longest run of consecutive identical values (1 if none repeat, 0 if empty)."""
    if len(frame_hash) == 0:
        return 0
    best = run = 1
    for i in range(1, len(frame_hash)):
        run = run + 1 if frame_hash[i] == frame_hash[i - 1] else 1
        best = max(best, run)
    return best


def dead_joints(state: np.ndarray, min_range: float = 1e-3) -> list[int]:
    """Joints whose range (max - min over the episode) is below min_range, while at least one other joint
    moves more than min_range (an arm that does not move at all is a different problem)."""
    ranges = state.max(axis=0) - state.min(axis=0)
    if not (ranges > min_range).any():
        return []
    return [j for j, r in enumerate(ranges) if r < min_range]


def limit_fraction(state: np.ndarray, limits: np.ndarray, margin: float = 0.02) -> float:
    """Fraction of frames where ANY joint is within `margin` of its lower or upper limit, or beyond it.
    limits: (J, 2) array of [low, high]."""
    low, high = limits[:, 0], limits[:, 1]
    hit = (state <= low + margin) | (state >= high - margin)
    return float(hit.any(axis=1).mean())


def robust_z(x: np.ndarray) -> np.ndarray:
    """0.6745 * (x - median) / MAD, MAD = median(|x - median|). If MAD == 0, return zeros."""
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0:
        return np.zeros_like(x, dtype=float)
    return 0.6745 * (x - med) / mad


def content_hash(ep: dict) -> str:
    """SHA-256 of the episode's state and action arrays (as float64, C order, bytes) and their shapes.
    Metadata (id, station, ...) is NOT part of it: two uploads of the same recording must collide."""
    h = hashlib.sha256()
    for k in ("state", "action"):
        a = np.ascontiguousarray(ep[k], dtype=np.float64)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def qa_report(eps: list[dict], limits: np.ndarray, fps: float, max_frozen_s: float = 2.0,
              max_limit_frac: float = 0.1, z_thresh: float = 3.5) -> dict:
    """Run every check and return {"quarantine": {episode id: sorted list of reasons}, "duplicates": sorted list of
    (kept id, duplicate id)}. Reasons (strings): "frozen_camera" (longest frozen run > max_frozen_s * fps frames),
    "dead_joint", "limits" (limit_fraction > max_limit_frac), "length_outlier" (|robust_z of lengths| > z_thresh),
    "duplicate" (same content hash as an EARLIER episode in the list; the first one is kept and not flagged).
    Only episodes with at least one reason appear in "quarantine"."""
    reasons: dict[int, set] = {}
    z = robust_z(np.array([len(e["state"]) for e in eps], dtype=float))
    seen: dict[str, int] = {}
    dups = []
    for i, e in enumerate(eps):
        r = set()
        if longest_frozen_run(e["frame_hash"]) > max_frozen_s * fps:
            r.add("frozen_camera")
        if dead_joints(e["state"]):
            r.add("dead_joint")
        if limit_fraction(e["state"], limits) > max_limit_frac:
            r.add("limits")
        if abs(z[i]) > z_thresh:
            r.add("length_outlier")
        h = content_hash(e)
        if h in seen:
            r.add("duplicate")
            dups.append((seen[h], e["id"]))
        else:
            seen[h] = e["id"]
        if r:
            reasons[e["id"]] = r
    return {"quarantine": {k: sorted(v) for k, v in sorted(reasons.items())}, "duplicates": sorted(dups)}
