"""REVIEW. SOLUTION.

BUG: A and B are resampled with different indices, which treats them as independent samples. They were run
on the same starts, so start difficulty is shared and cancels in the difference. Resample start indices
ONCE per bootstrap draw and use them for both policies (paired bootstrap): the CI shrinks to what the
discordant starts actually say.
"""
import random


def diff_ci(a: list[int], b: list[int], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(b[i] - a[i] for i in idx) / n)
    diffs.sort()
    return diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot) - 1]
