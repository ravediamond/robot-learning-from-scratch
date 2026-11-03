"""W2 D1: datasets and loaders for episodic robot data."""
import torch
from torch.utils.data import DataLoader, Dataset

Episode = dict[str, torch.Tensor]   # {"obs": (T, Do), "action": (T, Da)}


class EpisodeWindowDataset(Dataset):
    """One sample per frame of every episode.

    sample = {
        "obs":         (Do,)     obs at time t
        "actions":     (H, Da)   action[t : t+H]; past the episode end, repeat the LAST action
        "action_mask": (H,) bool True on real steps, False on padded ones
        "episode":     int       episode index (in the input list)
    }
    No Python list of all (episode, t) pairs: map the global index with cumulative lengths.
    """

    def __init__(self, episodes: list[Episode], H: int):
        # TODO
        raise NotImplementedError

    def __len__(self) -> int:
        # TODO
        raise NotImplementedError

    def __getitem__(self, i: int) -> dict:
        # TODO
        raise NotImplementedError


def collate_windows(samples: list[dict]) -> dict:
    """Stack each tensor field on a new dim 0; ints become a long tensor (B,). No default_collate."""
    # TODO
    raise NotImplementedError


def make_loader(ds: Dataset, batch_size: int, seed: int, shuffle: bool = True) -> DataLoader:
    """DataLoader with collate_windows and a reproducible shuffle (same seed -> same order)."""
    # TODO
    raise NotImplementedError


def task_balanced_weights(task_ids: torch.Tensor) -> torch.Tensor:
    """task_ids: (N,) long. Per-sample weights (N,) so every task has the same total weight,
    and the weights sum to 1."""
    # TODO
    raise NotImplementedError
