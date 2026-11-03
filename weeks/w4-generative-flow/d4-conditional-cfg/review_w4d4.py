"""REVIEW. SOLUTION.

BUG: the weights are swapped: (1-w) v_c + w v_u gives the unconditional field at w=1 and pushes AWAY
from the condition for w>1. Guidance is v_u + w (v_c - v_u) = (1-w) v_u + w v_c.
"""


def guided_velocity(model, x, t, cond, null_cond, w):
    v_c = model(x, t, cond)
    v_u = model(x, t, null_cond)
    return v_u + w * (v_c - v_u)
