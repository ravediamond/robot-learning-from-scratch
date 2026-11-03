"""REVIEW. SOLUTION.

BUG: a plain reshape merges r*r CONSECUTIVE tokens in row-major order (e.g. 4 tokens of the same row,
possibly wrapping to the next row), not an r x r spatial block. Reshape to the grid, permute, reshape.
"""
import torch


def merge_tokens(x, r=2):
    B, N, C = x.shape
    h = int(round(N ** 0.5))
    g = x.view(B, h // r, r, h // r, r, C).permute(0, 1, 3, 2, 4, 5)
    return g.reshape(B, N // (r * r), C * r * r)
