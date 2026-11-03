"""W1 D1: tensors and shapes. SOLUTION."""
import torch


def to_tensor(rows: list[list[float]]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.float32)  # (N, D)


def flatten_time(x: torch.Tensor) -> torch.Tensor:
    B, T, D = x.shape
    return x.reshape(B * T, D)  # reshape, not view: x may be non-contiguous


def unflatten_time(x: torch.Tensor, B: int, T: int) -> torch.Tensor:
    return x.reshape(B, T, x.shape[-1])  # (B, T, D)


def pad_episodes(episodes: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    N = len(episodes)
    T_max = max(e.shape[0] for e in episodes)
    D = episodes[0].shape[1]
    padded = torch.zeros(N, T_max, D, dtype=episodes[0].dtype)  # (N, T_max, D)
    mask = torch.zeros(N, T_max, dtype=torch.bool)  # (N, T_max)
    for i, e in enumerate(episodes):
        T_i = e.shape[0]
        padded[i, :T_i] = e
        mask[i, :T_i] = True
    return padded, mask
    # Built-in alternative: torch.nn.utils.rnn.pad_sequence(episodes, batch_first=True)


def normalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # mean/std (D,) broadcast against the last dim of x (..., D)
    return (x - mean) / (std + eps)


def action_chunks(actions: torch.Tensor, H: int) -> torch.Tensor:
    # unfold(0, H, 1) on (T, D) -> (T-H+1, D, H): the window dim goes LAST
    return actions.unfold(0, H, 1).permute(0, 2, 1)  # (T-H+1, H, D)


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
