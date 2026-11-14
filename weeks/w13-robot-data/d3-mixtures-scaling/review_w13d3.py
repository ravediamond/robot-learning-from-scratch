"""REVIEW. SOLUTION.

BUG: each item got its source's probability, so a source's total weight is p_source times its number of items:
the big public dataset wins again. Divide by the source's item count so that the total weight of each source
equals its target probability.
"""
import numpy as np


def sampling_weights(source_of_item: list[str], p_source: dict[str, float]) -> np.ndarray:
    counts: dict[str, int] = {}
    for s in source_of_item:
        counts[s] = counts.get(s, 0) + 1
    w = np.array([p_source[s] / counts[s] for s in source_of_item], dtype=float)
    return w / w.sum()
