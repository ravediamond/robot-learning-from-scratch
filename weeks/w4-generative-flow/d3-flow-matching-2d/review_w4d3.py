"""REVIEW. SOLUTION.

BUG: x_t uses the opposite convention (t=0 is data, t=1 is noise) while the target (actions - noise) and
the sampler assume t=0 is noise. The network learns velocities for the wrong points: sampling diverges.
"""
import torch


def make_fm_batch(actions, generator):
    noise = torch.randn(actions.shape, generator=generator)
    t = torch.rand(actions.shape[0], generator=generator)
    tt = t[:, None, None]
    x_t = (1 - tt) * noise + tt * actions
    return x_t, t, actions - noise
