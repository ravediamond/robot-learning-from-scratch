"""REVIEW. SOLUTION.

BUG: torch.max instead of torch.min. The clipped objective must be the PESSIMISTIC one: with max, the
clip is only active when it increases the objective, so nothing stops the ratio from running away.
"""
import torch


def ppo_loss(logp_new, logp_old, adv, clip=0.2):
    ratio = (logp_new - logp_old).exp()
    surr1 = ratio * adv
    surr2 = ratio.clamp(1 - clip, 1 + clip) * adv
    return -torch.min(surr1, surr2).mean()
