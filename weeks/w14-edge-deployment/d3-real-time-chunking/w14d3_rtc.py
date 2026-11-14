"""W14 D3: real-time chunking as guided inpainting on a flow sampler (1-D chunks)."""
import torch


def shift_previous(prev_chunk: torch.Tensor, executed: int) -> torch.Tensor:
    """Actions of the previous chunk that are still to come, aligned with the new chunk's index 0:
    prev_chunk[executed:], padded at the end by repeating its last action, to the same length."""
    # TODO
    raise NotImplementedError


def soft_mask(H: int, fixed: int, overlap: int) -> torch.Tensor:
    """Weights (H,): 1 for i < fixed; for fixed <= i < overlap a linear decay from 1 down towards 0:
    (overlap - i) / (overlap - fixed + 1); 0 for i >= overlap."""
    # TODO
    raise NotImplementedError


def guided_sample(velocity_fn, target: torch.Tensor, mask: torch.Tensor, steps: int, guidance: float,
                  generator: torch.Generator) -> torch.Tensor:
    """Euler sampling of a chunk from x0 = torch.randn(H, generator=generator), t_i = i / steps.
    velocity_fn(x, t) -> (v, x1_hat). Guided velocity: v + guidance * mask * (target - x1_hat).
    Return the final x."""
    # TODO
    raise NotImplementedError


def seam_jump(target: torch.Tensor, new_chunk: torch.Tensor, fixed: int) -> float:
    """At the switch, the robot goes from target[fixed - 1] (the last old action executed) to new_chunk[fixed]
    (the first new action used). Return |new_chunk[fixed] - target[fixed - 1]|."""
    # TODO
    raise NotImplementedError
