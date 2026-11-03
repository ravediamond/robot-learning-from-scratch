"""REVIEW. SOLUTION.

BUG: decodes to the LOWER EDGE of each bin, not its centre: every value is biased by -width/2 on every
joint (always "too low"). Decode to low + (idx + 0.5) * width.
"""
import torch


def detokenize(tokens, low, high, n_bins):
    width = (high - low) / n_bins
    return low + (tokens.float() + 0.5) * width
