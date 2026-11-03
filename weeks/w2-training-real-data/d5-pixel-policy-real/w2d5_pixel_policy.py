"""W2 D5: pixel + state chunk policy. You may use cnn_ref (or copy your own D3 encoder)."""
import torch
import torch.nn as nn

from cnn_ref import CNNEncoder, random_shift

Data = dict[str, torch.Tensor]   # img (N,3,S,S) [0,1], state (N,Ds), actions (N,H,Da), mask (N,H) bool


def masked_chunk_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """pred, target (N, H, D), mask (N, H). Mean over real (step, dim) entries only."""
    # TODO
    raise NotImplementedError


class PixelStatePolicy(nn.Module):
    """Raw state + image in, raw action chunk out (N, H, Da).

    Buffers: state_mean, state_std, act_mean, act_std (shapes (Ds,), (Ds,), (Da,), (Da,)).
    use_image=False: ignore the image entirely (no CNN), state-only twin with the same head.
    forward_normalized(img, state) returns the chunk in normalized action space.
    """

    def __init__(self, stats: dict[str, torch.Tensor], H: int, use_image: bool = True, hidden: int = 256):
        super().__init__()
        # TODO

    def forward_normalized(self, img: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        # TODO
        raise NotImplementedError

    def forward(self, img: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        # TODO
        raise NotImplementedError


def train_policy(data: Data, use_image: bool, epochs: int = 20, batch_size: int = 64, lr: float = 1e-3,
                 aug_pad: int = 2, seed: int = 0) -> PixelStatePolicy:
    """Stats from `data` (state over N; actions over N and H, real steps only), Adam, masked MSE in
    normalized action space, random_shift(img, aug_pad) on training batches only. Return in eval mode."""
    # TODO
    raise NotImplementedError


def evaluate(policy: PixelStatePolicy, data: Data) -> float:
    """Masked chunk MSE in RAW units, no grad, batched (chunks of 512)."""
    # TODO
    raise NotImplementedError
