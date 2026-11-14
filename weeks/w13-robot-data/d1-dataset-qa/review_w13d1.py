"""REVIEW. SOLUTION.

BUG: mean and standard deviation are themselves pulled by the outliers. One huge episode inflates the sd so
much that small outliers sit within 3 sd of a shifted mean: masking. Use robust statistics: the median and
the median absolute deviation (z = 0.6745 (x - median) / MAD, threshold 3.5).
"""
import numpy as np


def length_outliers(lengths: np.ndarray, thresh: float = 3.5) -> list[int]:
    med = np.median(lengths)
    mad = np.median(np.abs(lengths - med))
    if mad == 0:
        return []
    z = 0.6745 * (lengths - med) / mad
    return [i for i in range(len(lengths)) if abs(z[i]) > thresh]
