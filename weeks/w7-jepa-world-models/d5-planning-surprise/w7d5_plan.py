"""W7 D5: CEM planning in latent space, and surprise."""
import sys
from pathlib import Path

import torch

import pixel_world as W

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "d4-mini-lewm"))
from w7d4_lewm import Encoder, Predictor  # noqa: E402,F401


@torch.no_grad()
def cem_plan(pred, z0: torch.Tensor, cost_fn, H: int = 5, N: int = 200, K: int = 20, iters: int = 4,
             generator: torch.Generator | None = None) -> torch.Tensor:
    """z0 (E, D). cost_fn(z (E, N, D)) -> (E, N) cost of one predicted latent. Returns the first action (E, 2)
    of the final mean. Cost of a sequence = sum over the H predicted latents."""
    # TODO
    raise NotImplementedError


def latent_cost(z_goal: torch.Tensor):
    """z_goal (E, D) -> cost_fn(z (E, N, D)) = squared L2 distance to the goal latent."""
    # TODO
    raise NotImplementedError


def probe_cost(probe, goal_pos: torch.Tensor):
    """probe: module D -> 2 (decoded position); goal_pos (E, 2) -> cost_fn = squared distance of probe(z) to goal."""
    # TODO
    raise NotImplementedError


@torch.no_grad()
def run_planner(enc, pred, start: torch.Tensor, goal: torch.Tensor, make_cost, steps: int = 25, seed: int = 0,
                act_fn=None) -> torch.Tensor:
    """Closed loop in pixel_world. z_goal = enc(render(goal)); cost = make_cost(z_goal, goal).
    Each step: z = enc(render(pos)); a = cem_plan(pred, z, cost, generator=g) (or act_fn(z, g) if given);
    pos = W.step(pos, a). Returns the final positions (E, 2). g = Generator(seed)."""
    # TODO
    raise NotImplementedError


@torch.no_grad()
def surprise_scores(enc, pred, n_episodes: int = 50, jump: float = 0.6, seed: int = 3):
    """Episodes from W.make_episodes(n_episodes, 20, seed). At transition t = 10 (between frames 10 and 11) the
    point teleports: positions from frame 11 on are (pos + jump).clamp(-W.LIM, W.LIM); re-render them.
    Returns (scores (E*20,) = ||pred(z_t, a_t) - z_{t+1}||^2, labels (E*20,) bool True on the teleport transition)."""
    # TODO
    raise NotImplementedError
