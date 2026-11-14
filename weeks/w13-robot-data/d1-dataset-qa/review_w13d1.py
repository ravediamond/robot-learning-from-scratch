"""REVIEW. Length-outlier check of the upload pipeline. A batch with one 40-minute episode (recording left on)
passed with only that episode flagged; three 2-second mis-clicks in the same batch went to training.
Find the bug, fix, replace:
BUG:
"""
import numpy as np


def length_outliers(lengths: np.ndarray, thresh: float = 3.0) -> list[int]:
    z = (lengths - lengths.mean()) / lengths.std()
    return [i for i in range(len(lengths)) if abs(z[i]) > thresh]
