"""W13 D2: near-duplicates, group-aware splits, dataset fingerprint, removal ledger."""
import hashlib
import random

import numpy as np


def resample(traj: np.ndarray, n: int) -> np.ndarray:
    """Linear interpolation of a (T, D) trajectory to n points evenly spaced in normalised time [0, 1]
    (np.interp per dimension)."""
    T = len(traj)
    src = np.linspace(0, 1, T)
    dst = np.linspace(0, 1, n)
    return np.stack([np.interp(dst, src, traj[:, d]) for d in range(traj.shape[1])], axis=1)


def near_duplicates(trajs: list[np.ndarray], n: int = 50, tol: float = 0.05) -> list[tuple[int, int]]:
    """Pairs (i, j), i < j, whose resampled trajectories have an RMS difference (sqrt of the mean over points
    and dimensions of the squared difference) below tol. Sorted."""
    rs = np.stack([resample(t, n) for t in trajs])          # (N, n, D)
    out = []
    for i in range(len(rs)):
        d = np.sqrt(((rs[i + 1:] - rs[i]) ** 2).mean(axis=(1, 2)))
        out += [(i, i + 1 + k) for k in np.where(d < tol)[0]]
    return sorted(out)


def group_split(groups: list[str], val_frac: float, seed: int) -> tuple[list[int], list[int]]:
    """Split item indices so that every group is entirely in train or entirely in validation.
    Unique groups sorted, shuffled with random.Random(seed); take groups for validation in that order until
    the validation items reach at least val_frac of all items. Return (train indices, val indices), each sorted."""
    uniq = sorted(set(groups))
    random.Random(seed).shuffle(uniq)
    target = val_frac * len(groups)
    val_groups, count = set(), 0
    for g in uniq:
        if count >= target:
            break
        val_groups.add(g)
        count += groups.count(g)
    va = [i for i, g in enumerate(groups) if g in val_groups]
    tr = [i for i, g in enumerate(groups) if g not in val_groups]
    return tr, va


def dataset_fingerprint(episode_hashes: list[str]) -> str:
    """SHA-256 of the SORTED episode hashes joined with newlines: independent of order, sensitive to content."""
    return hashlib.sha256("\n".join(sorted(episode_hashes)).encode()).hexdigest()


class Ledger:
    """Every removal decision, append only. Entries: (episode id, rule, version)."""

    def __init__(self):
        self.entries: list[tuple[int, str, str]] = []

    def remove(self, episode_id: int, rule: str, version: str) -> None:
        self.entries.append((episode_id, rule, version))

    def restore(self, episode_id: int, version: str) -> None:
        """Record a restore as an entry with rule "restore" (nothing is ever deleted from the ledger)."""
        self.entries.append((episode_id, "restore", version))

    def removed(self) -> set[int]:
        """Ids whose LAST entry is a removal."""
        last: dict[int, str] = {}
        for eid, rule, _ in self.entries:
            last[eid] = rule
        return {e for e, r in last.items() if r != "restore"}

    def apply(self, ids: list[int]) -> list[int]:
        """The ids that survive curation, in the given order."""
        gone = self.removed()
        return [i for i in ids if i not in gone]
