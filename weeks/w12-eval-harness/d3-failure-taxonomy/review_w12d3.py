"""REVIEW. Failure chart of the weekly eval report ("% of failures by cause"). It said the new policy has a
"severe grasp problem" (100% of failures) while it fails grasps 5 times in 50 against 10 for the baseline.
Find the bug, fix, replace:
BUG:
"""
from collections import Counter


def cause_rates(records: list[dict], policy: str) -> dict[str, float]:
    fails = [r["failure_tag"] for r in records if r["policy"] == policy and r["outcome"] == "failure"]
    counts = Counter(fails)
    return {tag: c / len(fails) for tag, c in counts.items()}
