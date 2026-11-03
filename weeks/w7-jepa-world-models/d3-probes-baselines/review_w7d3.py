"""REVIEW. SOLUTION.

BUG: the denominator uses the variance of the PREDICTIONS instead of the TARGETS. A probe that predicts
almost constant values has a tiny denominator... and wildly wrong R² (here: huge positive or negative);
the reference is always the variance of Y around its mean.
"""
import torch


def r_squared(pred, Y):
    ss_res = ((Y - pred) ** 2).sum()
    ss_tot = ((Y - Y.mean(0)) ** 2).sum()
    return (1 - ss_res / ss_tot).item()
