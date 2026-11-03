"""W1 D5: behavior cloning on toy SO-101 data. Uses so101_toy.generate_episodes."""
import torch
import torch.nn as nn

Episode = dict[str, torch.Tensor]   # {"obs": (T, 12), "action": (T, 6)}


def split_by_episode(n_episodes: int, val_frac: float = 0.2, seed: int = 0) -> tuple[list[int], list[int]]:
    """Shuffle episode indices with the seed; the first round(val_frac * n) go to val.

    Returns (train_ids, val_ids), disjoint, together covering range(n_episodes).
    """
    # TODO
    raise NotImplementedError


def stack_frames(episodes: list[Episode], ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate the frames of the selected episodes -> obs (N, 12), action (N, 6)."""
    # TODO
    raise NotImplementedError


def compute_norm_stats(obs: torch.Tensor, act: torch.Tensor, min_std: float = 1e-2) -> dict[str, torch.Tensor]:
    """Per-dim mean/std. Keys: obs_mean, obs_std, act_mean, act_std. Clamp std to >= min_std."""
    # TODO
    raise NotImplementedError


class BCPolicy(nn.Module):
    """MLP policy that takes RAW obs (..., 12) and returns RAW actions (..., 6).

    Internally: normalize obs with obs stats -> MLP -> un-normalize with act stats.
    Store the four stats as buffers so they are saved and moved with the model.
    Expose `forward_normalized(obs_raw)` returning the MLP output in normalized action space
    (used for the loss).
    """

    def __init__(self, stats: dict[str, torch.Tensor], hidden: int = 256):
        super().__init__()
        # TODO

    def forward_normalized(self, obs: torch.Tensor) -> torch.Tensor:
        # TODO
        raise NotImplementedError

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # TODO
        raise NotImplementedError


def train_bc(obs: torch.Tensor, act: torch.Tensor, epochs: int = 40, batch_size: int = 256,
             lr: float = 1e-3, seed: int = 0) -> BCPolicy:
    """Compute stats on (obs, act), build a BCPolicy, train with Adam on MSE in NORMALIZED
    action space, shuffled mini-batches. Return the policy in eval mode."""
    # TODO
    raise NotImplementedError


def raw_mse(policy: nn.Module, obs: torch.Tensor, act: torch.Tensor) -> float:
    """MSE in raw action units, no graph."""
    # TODO
    raise NotImplementedError


def hold_baseline_mse(obs: torch.Tensor, act: torch.Tensor) -> float:
    """MSE in raw units of the policy 'action = current joint state' (obs[:, :6])."""
    # TODO
    raise NotImplementedError


def save_policy(policy: BCPolicy, path: str, hidden: int = 256) -> None:
    """Save everything needed to rebuild the policy: state_dict + hidden size."""
    # TODO
    raise NotImplementedError


def load_policy(path: str) -> BCPolicy:
    """Rebuild a BCPolicy and load its weights STRICTLY. The stats must come from the file.
    Hint: build it with dummy stats of the right shapes, then load_state_dict(strict=True)."""
    # TODO
    raise NotImplementedError
