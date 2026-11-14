"""W13 D2: near-duplicates, group-aware splits, dataset fingerprint, removal ledger."""
import hashlib
import random

import numpy as np


def resample(traj: np.ndarray, n: int) -> np.ndarray:
    """Linear interpolation of a (T, D) trajectory to n points evenly spaced in normalised time [0, 1]
    (np.interp per dimension)."""
    # TODO
    raise NotImplementedError


def near_duplicates(trajs: list[np.ndarray], n: int = 50, tol: float = 0.05) -> list[tuple[int, int]]:
    """Pairs (i, j), i < j, whose resampled trajectories have an RMS difference (sqrt of the mean over points
    and dimensions of the squared difference) below tol. Sorted."""
    # TODO
    raise NotImplementedError


def group_split(groups: list[str], val_frac: float, seed: int) -> tuple[list[int], list[int]]:
    """Split item indices so that every group is entirely in train or entirely in validation.
    Unique groups sorted, shuffled with random.Random(seed); take groups for validation in that order until
    the validation items reach at least val_frac of all items. Return (train indices, val indices), each sorted."""
    # TODO
    raise NotImplementedError


def dataset_fingerprint(episode_hashes: list[str]) -> str:
    """SHA-256 of the SORTED episode hashes joined with newlines: independent of order, sensitive to content."""
    # TODO
    raise NotImplementedError


class Ledger:
    """Every removal decision, append only. Entries: (episode id, rule, version)."""

    def __init__(self):
        self.entries: list[tuple[int, str, str]] = []

    def remove(self, episode_id: int, rule: str, version: str) -> None:
        # TODO
        raise NotImplementedError

    def restore(self, episode_id: int, version: str) -> None:
        """Record a restore as an entry with rule "restore" (nothing is ever deleted from the ledger)."""
        # TODO
        raise NotImplementedError

    def removed(self) -> set[int]:
        """Ids whose LAST entry is a removal."""
        # TODO
        raise NotImplementedError

    def apply(self, ids: list[int]) -> list[int]:
        """The ids that survive curation, in the given order."""
        # TODO
        raise NotImplementedError
