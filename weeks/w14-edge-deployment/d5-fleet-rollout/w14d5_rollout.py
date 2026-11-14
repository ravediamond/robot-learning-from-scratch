"""W14 D5: artifact verification, shadow comparison, canary statistics, staged rollout."""
import hashlib
import math

import numpy as np
import torch


def fingerprint(state_dict: dict) -> str:
    """Provided: same definition as W12 D5."""
    h = hashlib.sha256()
    for key in sorted(state_dict):
        t = state_dict[key].detach().cpu().contiguous()
        h.update(key.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        if t.dtype == torch.bfloat16:
            t = t.view(torch.int16)
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def verify_artifact(state_dict: dict, approved: str) -> bool:
    """True iff the loaded weights' fingerprint equals the approved one."""
    return fingerprint(state_dict) == approved


def shadow_disagreement(deployed: np.ndarray, candidate: np.ndarray, tol: float) -> dict:
    """deployed, candidate: (T, J) actions on the same observations. A step disagrees if any joint differs by more
    than tol. Return {"rate": share of disagreeing steps, "steps": list of their indices, "max_abs": max |diff|}."""
    diff = np.abs(deployed - candidate)
    bad = (diff > tol).any(axis=1)
    return {"rate": float(bad.mean()), "steps": [int(i) for i in np.where(bad)[0]], "max_abs": float(diff.max())}


def rate_ratio_pvalue(k_canary: int, h_canary: float, k_base: int, h_base: float) -> float:
    """One-sided test that the canary's event rate is HIGHER than the baseline's. Under equal rates, given the
    total n = k_canary + k_base, k_canary ~ Binomial(n, p0) with p0 = h_canary / (h_canary + h_base).
    Return P(X >= k_canary) (1.0 when n == 0)."""
    n = k_canary + k_base
    if n == 0:
        return 1.0
    p0 = h_canary / (h_canary + h_base)
    return sum(math.comb(n, x) * p0 ** x * (1 - p0) ** (n - x) for x in range(k_canary, n + 1))


def canary_decision(k_canary: int, h_canary: float, k_base: int, h_base: float, min_hours: float = 20.0,
                    alpha: float = 0.05) -> str:
    """"wait" if h_canary < min_hours; "rollback" if rate_ratio_pvalue < alpha; else "continue"."""
    if h_canary < min_hours:
        return "wait"
    return "rollback" if rate_ratio_pvalue(k_canary, h_canary, k_base, h_base) < alpha else "continue"


def stage_sizes(n_robots: int, fractions: list[float]) -> list[int]:
    """Robots running the new policy at each stage: ceil(fraction * n_robots), at least 1, non-decreasing."""
    out, prev = [], 0
    for f in fractions:
        k = max(1, math.ceil(f * n_robots - 1e-9), prev)
        out.append(k)
        prev = k
    return out
