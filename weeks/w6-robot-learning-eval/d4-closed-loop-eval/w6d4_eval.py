"""W6 D4: closed-loop evaluation harness on sim2d."""
import torch

import sim2d


def fixed_starts(n: int, seed: int) -> torch.Tensor:
    """(n, 2) start states from sim2d.sample_starts with Generator(seed). Same seed -> same starts."""
    # TODO
    raise NotImplementedError


@torch.no_grad()
def evaluate(act_fn, starts: torch.Tensor, H_exec: int, seed: int = 0) -> dict:
    """Roll out act_fn(obs (k, 2), generator) -> raw chunks (k, H, 2) from the GIVEN starts (do not modify
    them). Generator(seed) for the policy's sampling. Replan every H_exec steps, clip actions, collision then
    success checks, stop finished episodes, T_MAX steps total.
    Returns {"success": (n,) bool, "collision": (n,) bool, "steps": (n,) long}."""
    # TODO
    raise NotImplementedError


def paired_counts(success_a: torch.Tensor, success_b: torch.Tensor) -> dict:
    """{"n11": both, "n10": A only, "n01": B only, "n00": neither} as ints."""
    # TODO
    raise NotImplementedError


@torch.no_grad()
def val_chunk_mse(act_fn, demos: dict, seed: int = 0) -> float:
    """Mean squared error (raw units, masked over real steps) between ONE sample of act_fn(demos['obs'])
    and demos['chunks']."""
    # TODO
    raise NotImplementedError
