"""W2 D1: SOLUTION."""
import torch
from torch.utils.data import DataLoader, Dataset

Episode = dict[str, torch.Tensor]


class EpisodeWindowDataset(Dataset):
    def __init__(self, episodes: list[Episode], H: int):
        self.episodes, self.H = episodes, H
        lengths = torch.tensor([len(e["obs"]) for e in episodes])
        self.ends = torch.cumsum(lengths, 0)                 # (E,) exclusive end of each episode
        self.starts = self.ends - lengths

    def __len__(self) -> int:
        return int(self.ends[-1])

    def __getitem__(self, i: int) -> dict:
        e = int(torch.searchsorted(self.ends, torch.tensor(i), right=True))
        t = i - int(self.starts[e])
        ep = self.episodes[e]
        T = len(ep["obs"])
        steps = torch.arange(t, t + self.H)                  # (H,)
        mask = steps < T
        actions = ep["action"][steps.clamp(max=T - 1)]       # (H, Da), repeats the last action
        return {"obs": ep["obs"][t], "actions": actions, "action_mask": mask, "episode": e}


def collate_windows(samples: list[dict]) -> dict:
    out = {}
    for k in samples[0]:
        vals = [s[k] for s in samples]
        out[k] = torch.stack(vals) if torch.is_tensor(vals[0]) else torch.tensor(vals, dtype=torch.long)
    return out


def make_loader(ds: Dataset, batch_size: int, seed: int, shuffle: bool = True) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_windows,
                      generator=torch.Generator().manual_seed(seed))


def task_balanced_weights(task_ids: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(task_ids)                         # (K,)
    n_tasks = (counts > 0).sum()
    return 1.0 / (counts[task_ids].float() * n_tasks)
