"""REVIEW. A faster window lookup: all episodes are concatenated into one flat (N, Da)
action tensor (like a LeRobot parquet file), and windows are sliced from it directly.
Training loss looked normal. Find the bug, fix it, replace this line:
BUG:
"""
import torch


def window_actions(actions: torch.Tensor, episode_index: torch.Tensor, t: int, H: int) -> tuple[torch.Tensor, torch.Tensor]:
    """actions: (N, Da) flat over all episodes. episode_index: (N,) episode of each frame (sorted).
    Returns (H, Da) actions starting at global frame t, and a (H,) bool mask of real steps.
    Steps past the end of t's episode repeat its last action and are masked out.
    """
    N = actions.shape[0]
    idx = torch.arange(t, t + H).clamp(max=N - 1)    # stay inside the tensor
    mask = torch.arange(t, t + H) < N
    return actions[idx], mask
