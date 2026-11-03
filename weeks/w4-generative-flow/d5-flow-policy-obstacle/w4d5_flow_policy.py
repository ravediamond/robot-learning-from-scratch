"""W4 D5: MSE vs flow-matching chunk policies in the obstacle world. Convention t=0 noise, t=1 data."""
import torch
import torch.nn as nn

import sim2d
from toy2d import time_embedding


class MSEChunkPolicy(nn.Module):
    """obs (N, 2) -> MLP (256, 256, ReLU) -> (N, H, 2) actions in units of MAX_STEP."""

    def __init__(self, H: int, hidden: int = 256):
        super().__init__()
        # TODO

    def forward(self, obs):
        # TODO
        raise NotImplementedError

    @torch.no_grad()
    def act(self, obs, generator=None):
        """Raw actions (N, H, 2) (multiply by MAX_STEP)."""
        # TODO
        raise NotImplementedError


class FlowChunkPolicy(nn.Module):
    """v(x_t, t, obs): concat(flatten x_t (H*2), time_embedding(t, 32), obs (2)) -> MLP (3 x 256, SiLU) -> (N, H, 2)."""

    def __init__(self, H: int, hidden: int = 256):
        super().__init__()
        # TODO

    def forward(self, x, t, obs):
        # TODO
        raise NotImplementedError

    @torch.no_grad()
    def act(self, obs, generator=None, steps: int = 10):
        """Euler from noise (N, H, 2) at t=0 to t=1, then multiply by MAX_STEP."""
        # TODO
        raise NotImplementedError


def masked_mse(pred, target, mask):
    """(N, H, 2), (N, H, 2), (N, H) bool -> scalar mean over real steps."""
    # TODO
    raise NotImplementedError


def train_mse_policy(demos: dict, H: int, steps: int = 3000, batch: int = 256, lr: float = 1e-3, seed: int = 0):
    # TODO
    raise NotImplementedError


def train_flow_policy(demos: dict, H: int, steps: int = 4000, batch: int = 256, lr: float = 1e-3, seed: int = 0):
    # TODO
    raise NotImplementedError


@torch.no_grad()
def rollout(act_fn, n: int, H_exec: int, seed: int = 0) -> dict:
    """act_fn(obs (k, 2), generator) -> raw action chunks (k, H, 2).

    n episodes in parallel from sim2d.sample_starts(n, Generator(seed)) (use that same generator for
    act_fn). Replan every H_exec steps for the still-active episodes. Each step: clip_action, move,
    mark collision (collides) then success (reached and not collided). Stop an episode at the first
    collision or success; stop everything after sim2d.T_MAX steps.
    Returns {"success": (n,) bool, "collision": (n,) bool, "steps": (n,) long}."""
    # TODO
    raise NotImplementedError
