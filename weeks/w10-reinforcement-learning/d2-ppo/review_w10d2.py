"""REVIEW. PPO loss from a teammate's branch. Training is unstable: the policy takes huge steps and the
return collapses after a few iterations. Find the bug, fix, replace:
BUG:
"""
import torch


def ppo_loss(logp_new, logp_old, adv, clip=0.2):
    ratio = (logp_new - logp_old).exp()
    surr1 = ratio * adv
    surr2 = ratio.clamp(1 - clip, 1 + clip) * adv
    return -torch.max(surr1, surr2).mean()
