"""REVIEW. SOLUTION.

BUG: the p99 is computed as mean + 2.33 standard deviations, which assumes a Gaussian. Latency has a long,
lumpy tail (garbage collection, thermal throttling, a slow frame): here 5% of calls take 200 ms and the
"p99" lands between the modes. Use the empirical percentile (nearest rank on the sorted samples), after
dropping warm-up calls.
"""
import math


def p99(samples_ms: list[float]) -> float:
    xs = sorted(samples_ms)
    return xs[math.ceil(0.99 * len(xs)) - 1]
