"""REVIEW. SOLUTION.

BUG: var(dim=1) is the spread of the entries INSIDE each vector, not the spread of each dimension ACROSS the
batch. A collapsed batch (all samples identical) can have spread-out entries and pays nothing. Use dim=0.
"""
import torch


def variance_term(z, eps=1e-4):
    return torch.relu(1 - torch.sqrt(z.var(dim=0) + eps)).mean()
