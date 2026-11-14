"""W13 D5: interventions to corrections, failure mining, collection planning."""
import math

import numpy as np


def intervention_segments(flags: np.ndarray, context: int) -> list[tuple[int, int]]:
    """flags[t] = 1 while a human controls the arm. For each maximal run of ones [s, e), return
    (max(0, s - context), e). Segments in order; overlapping or touching segments are merged."""
    segs = []
    t, T = 0, len(flags)
    while t < T:
        if flags[t]:
            s = t
            while t < T and flags[t]:
                t += 1
            segs.append([max(0, s - context), t])
        else:
            t += 1
    merged: list[list[int]] = []
    for s, e in segs:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [tuple(x) for x in merged]


def correction_dataset(log: dict, context: int) -> list[dict]:
    """log: {"obs": (T, D), "executed_action": (T, A), "policy_action": (T, A), "intervention": (T,) 0/1}.
    One sample per segment: {"obs": obs[s:e], "action": executed_action[s:e], "segment": (s, e)}."""
    out = []
    for s, e in intervention_segments(log["intervention"], context):
        out.append({"obs": log["obs"][s:e], "action": log["executed_action"][s:e], "segment": (s, e)})
    return out


def review_queue(episodes: list[dict], k: int, w_uncertainty: float = 0.5) -> list[int]:
    """episodes: [{"id", "failure_score" in [0, 1], "uncertainty" in [0, 1]}]. Priority =
    failure_score + w_uncertainty * uncertainty. Return the ids of the top k, highest first (ties: smaller id)."""
    ranked = sorted(episodes, key=lambda e: (-(e["failure_score"] + w_uncertainty * e["uncertainty"]), e["id"]))
    return [e["id"] for e in ranked[:k]]


def allocate_collection(failure_rate: list[float], n_eval: list[int], budget: int) -> list[int]:
    """Split `budget` demonstrations across conditions. Score_i = failure_rate_i + sqrt(p(1-p)/n) with
    p = failure_rate_i, n = n_eval_i (failures plus their uncertainty). Allocate by largest remainder:
    quota_i = budget * score_i / sum(score); floor every quota, then give the remaining units to the largest
    fractional parts (ties: lower index). Conditions with score 0 get 0."""
    score = [p + math.sqrt(p * (1 - p) / n) for p, n in zip(failure_rate, n_eval)]
    total = sum(score)
    quota = [budget * s / total for s in score]
    alloc = [math.floor(q) for q in quota]
    rest = budget - sum(alloc)
    order = sorted(range(len(quota)), key=lambda i: (-(quota[i] - alloc[i]), i))
    for i in order[:rest]:
        alloc[i] += 1
    return alloc
