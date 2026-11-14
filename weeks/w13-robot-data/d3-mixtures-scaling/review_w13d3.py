"""REVIEW. Weighted sampler for a 3-source mixture with target probabilities {public: 0.4, ours: 0.4, sim: 0.2}.
The training logs show 'ours' in barely 4% of batches. Find the bug, fix, replace:
BUG:
"""
import numpy as np


def sampling_weights(source_of_item: list[str], p_source: dict[str, float]) -> np.ndarray:
    w = np.array([p_source[s] for s in source_of_item], dtype=float)
    return w / w.sum()
