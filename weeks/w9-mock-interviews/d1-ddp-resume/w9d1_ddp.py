"""W9 D1: DDP with gloo on CPU, and exact resume. Functions must be top-level (spawned processes pickle them)."""
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def make_model(seed: int = 0) -> nn.Module:
    """PROVIDED. Same initial weights for every call with the same seed."""
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(8, 32), nn.Tanh(), nn.Linear(32, 2))


def make_data(n: int = 256, seed: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """PROVIDED."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 8, generator=g)
    return x, torch.stack([x[:, 0] * x[:, 1], x[:, 2].sin()], 1)


def global_batches(n: int, batch: int, steps: int, seed: int = 2) -> list[torch.Tensor]:
    """PROVIDED. The SAME list of global index batches for every rank and for the reference."""
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, n, (batch,), generator=g) for _ in range(steps)]


def ddp_worker(rank: int, world: int, port: int, steps: int, batch: int, lr: float, out_path: str) -> None:
    """Spawned once per rank. init_process_group("gloo", rank, world_size) with MASTER_ADDR=127.0.0.1 and
    MASTER_PORT=port; model = DDP(make_model(0)); SGD(lr); for each global batch idx (global_batches),
    this rank uses idx[rank::world]; MSE loss (mean); zero_grad, backward, step. Rank 0 saves
    model.module.state_dict() to out_path. destroy_process_group at the end."""
    # TODO
    raise NotImplementedError


def single_process_reference(steps: int, batch: int, lr: float) -> dict:
    """Same model, same global batches, ALL indices of each batch in one process. Returns the state_dict."""
    # TODO
    raise NotImplementedError


def save_checkpoint(path: str, model: nn.Module, opt: torch.optim.Optimizer, sched, step: int, gen: torch.Generator) -> None:
    """Everything needed to continue exactly: model, optimizer, scheduler, step, data-generator state."""
    # TODO
    raise NotImplementedError


def load_checkpoint(path: str, model: nn.Module, opt: torch.optim.Optimizer, sched, gen: torch.Generator) -> int:
    """Restore in place; return the step to continue from."""
    # TODO
    raise NotImplementedError


def train_steps(model, opt, sched, gen: torch.Generator, start: int, end: int, data) -> None:
    """PROVIDED-shape loop you complete: for step in [start, end): batch of 32 indices drawn with `gen`,
    MSE loss, zero_grad, backward, opt.step(), sched.step()."""
    # TODO
    raise NotImplementedError
