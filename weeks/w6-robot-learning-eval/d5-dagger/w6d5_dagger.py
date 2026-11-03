"""W6 D5: behaviour cloning vs DAgger under disturbances. Single-step policies (no chunks)."""
import torch
import torch.nn as nn

import dagger_world as w
import sim2d


def train_bc(obs: torch.Tensor, act: torch.Tensor, steps: int = 1500, seed: int = 0) -> nn.Module:
    """MLP 2 -> 128 -> 128 -> 2 (ReLU) predicting act / MAX_STEP, MSE, Adam 1e-3, batch 256. Eval mode."""
    # TODO
    raise NotImplementedError


@torch.no_grad()
def run(policy: nn.Module, n: int, seed: int, record: bool = False) -> dict:
    """n episodes from sim2d.sample_starts(n, Generator(seed)); the SAME generator then draws
    w.pushes(n, g). Each step t < T_MAX: if t == w.KICK_T add the push to the live episodes; if record,
    store the positions of the live episodes; action = clip_action(policy(pos) * MAX_STEP); move live
    episodes; collision, then success checks.
    Returns {"success", "collision"} (n,) bool and, if record, "visited" (V, 2)."""
    # TODO
    raise NotImplementedError


def aggregate(obs, act, visited) -> tuple[torch.Tensor, torch.Tensor]:
    """Append the visited states, labelled with the EXPERT's actions."""
    # TODO
    raise NotImplementedError


def dagger(obs, act, iterations: int, n_rollouts: int = 50, seed: int = 100) -> tuple[nn.Module, int]:
    """Start from train_bc(obs, act). Each iteration i: run(policy, n_rollouts, seed + i, record=True),
    aggregate, retrain with train_bc(..., seed=i + 1). Returns (final policy, dataset size)."""
    # TODO
    raise NotImplementedError
