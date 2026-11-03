"""W1 D1: tensors and shapes. Fill every TODO. Comment every tensor line with its shape."""
import torch


def to_tensor(rows: list[list[float]]) -> torch.Tensor:
    """List of rows -> float32 tensor of shape (len(rows), len(rows[0]))."""
    # TODO
    raise NotImplementedError


def flatten_time(x: torch.Tensor) -> torch.Tensor:
    """(B, T, D) -> (B*T, D). Must work even if x is not contiguous."""
    # TODO
    raise NotImplementedError


def unflatten_time(x: torch.Tensor, B: int, T: int) -> torch.Tensor:
    """(B*T, D) -> (B, T, D). Inverse of flatten_time."""
    # TODO
    raise NotImplementedError


def pad_episodes(episodes: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Episodes of shape (T_i, D) with different T_i.

    Returns:
        padded: (N, T_max, D) float tensor, zeros after each episode ends
        mask:   (N, T_max) bool tensor, True on real steps, False on padding
    """
    # TODO
    raise NotImplementedError


def normalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-feature z-score. x: (..., D), mean and std: (D,). Returns same shape as x.

    eps protects joints that never move (std == 0), e.g. a gripper that stayed open.
    """
    # TODO
    raise NotImplementedError


def action_chunks(actions: torch.Tensor, H: int) -> torch.Tensor:
    """All windows of H consecutive actions.

    actions: (T, D)  ->  chunks: (T - H + 1, H, D), with chunks[t] == actions[t : t + H].
    Hint: Tensor.unfold, then check where it puts the window dimension.
    """
    # TODO
    raise NotImplementedError


def pick_device() -> torch.device:
    """mps if available, else cuda if available, else cpu."""
    # TODO
    raise NotImplementedError
