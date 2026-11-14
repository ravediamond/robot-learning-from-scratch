"""REVIEW. Confidence interval of "success(B) - success(A)" in the team's eval dashboard. The two policies
were run on the same 50 start configurations, and the CI is so wide that no change ever looks real.
Find the bug, fix, replace:
BUG:
"""
import random


def diff_ci(a: list[int], b: list[int], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(n_boot):
        ia = [rng.randrange(n) for _ in range(n)]
        ib = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(b[i] for i in ib) / n - sum(a[i] for i in ia) / n)
    diffs.sort()
    return diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot) - 1]
