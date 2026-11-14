"""W12 D3: rater agreement, per-cause failure rates, stage funnel.

A record here: {"policy", "outcome" in OUTCOMES, "failure_tag" (None unless failure), "stage_reached": number of
STAGES completed, 0..len(STAGES)}."""
from collections import Counter

from evalkit import STAGES, TAXONOMY, wilson


def cohen_kappa(r1: list[str], r2: list[str]) -> float:
    """kappa = (po - pe) / (1 - pe). po: observed agreement. pe: sum over labels of P(r1 = label) * P(r2 = label).
    If pe == 1 (both raters always use the same single label), return 1.0."""
    n = len(r1)
    po = sum(a == b for a, b in zip(r1, r2)) / n
    c1, c2 = Counter(r1), Counter(r2)
    pe = sum(c1[k] * c2[k] for k in set(c1) | set(c2)) / (n * n)
    if pe == 1:
        return 1.0
    return (po - pe) / (1 - pe)


def breakdown(records: list[dict], policy: str) -> dict[str, dict]:
    """For every tag in TAXONOMY: {"count", "rate", "lo", "hi"} where rate = count / n and (lo, hi) = wilson(count, n),
    n = episodes of this policy whose outcome is "success" or "failure" (aborted episodes are excluded).
    Tags that never occur appear with count 0."""
    eps = [r for r in records if r["policy"] == policy and r["outcome"] in ("success", "failure")]
    n = len(eps)
    counts = Counter(r["failure_tag"] for r in eps if r["outcome"] == "failure")
    out = {}
    for tag in TAXONOMY:
        lo, hi = wilson(counts[tag], n)
        out[tag] = {"count": counts[tag], "rate": counts[tag] / n if n else 0.0, "lo": lo, "hi": hi}
    return out


def stage_funnel(records: list[dict], policy: str) -> list[dict]:
    """One row per stage k (in STAGES order): {"stage", "reached", "conditional", "cumulative"}.
    reached = episodes (success or failure, of this policy) with stage_reached >= k + 1.
    conditional = reached / reached of the previous stage (the first stage divides by n); None if that is 0.
    cumulative = reached / n."""
    eps = [r for r in records if r["policy"] == policy and r["outcome"] in ("success", "failure")]
    n = len(eps)
    rows, prev = [], n
    for k, stage in enumerate(STAGES):
        reached = sum(1 for r in eps if r["stage_reached"] >= k + 1)
        rows.append({"stage": stage, "reached": reached,
                     "conditional": reached / prev if prev else None,
                     "cumulative": reached / n if n else 0.0})
        prev = reached
    return rows


def biggest_drop(funnel: list[dict]) -> str:
    """The stage with the lowest conditional success (ignore None; ties: the earliest stage)."""
    valid = [(row["conditional"], i, row["stage"]) for i, row in enumerate(funnel) if row["conditional"] is not None]
    return min(valid)[2]
