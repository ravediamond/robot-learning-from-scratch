"""REVIEW. SOLUTION.

BUG: the target is obs[:, :-1] (the SAME frame as the input), not obs[:, 1:]. The predictor learns the
identity (z_hat = z_t): perfect loss, zero dynamics, and it can never beat copy-last, which it IS.
"""


def make_pairs(obs, action):
    return obs[:, :-1].flatten(0, 1), action.flatten(0, 1), obs[:, 1:].flatten(0, 1)
