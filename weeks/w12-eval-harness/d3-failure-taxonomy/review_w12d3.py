"""REVIEW. SOLUTION.

BUG: each cause was divided by the policy's number of FAILURES. A policy that rarely fails gets large shares
for its few failures, and two policies' shares cannot be compared. Divide by the policy's number of
(non-aborted) episodes: "grasp failures per rollout", which is what you compare and what you put a CI on.
"""
from collections import Counter


def cause_rates(records: list[dict], policy: str) -> dict[str, float]:
    eps = [r for r in records if r["policy"] == policy and r["outcome"] in ("success", "failure")]
    counts = Counter(r["failure_tag"] for r in eps if r["outcome"] == "failure")
    return {tag: c / len(eps) for tag, c in counts.items()}
