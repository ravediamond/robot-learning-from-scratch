"""W9 D1: SOLUTION."""
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def make_model(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(8, 32), nn.Tanh(), nn.Linear(32, 2))


def make_data(n=256, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 8, generator=g)
    return x, torch.stack([x[:, 0] * x[:, 1], x[:, 2].sin()], 1)


def global_batches(n, batch, steps, seed=2):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, n, (batch,), generator=g) for _ in range(steps)]


def ddp_worker(rank, world, port, steps, batch, lr, out_path):
    os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "127.0.0.1", str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    model = DDP(make_model(0))
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    x, y = make_data()
    for idx in global_batches(len(x), batch, steps):
        mine = idx[rank::world]
        loss = ((model(x[mine]) - y[mine]) ** 2).mean()
        opt.zero_grad()
        loss.backward()                                  # gradients all-reduced (averaged) here
        opt.step()
    if rank == 0:
        torch.save(model.module.state_dict(), out_path)
    dist.barrier()
    dist.destroy_process_group()


def single_process_reference(steps, batch, lr):
    model = make_model(0)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    x, y = make_data()
    for idx in global_batches(len(x), batch, steps):
        loss = ((model(x[idx]) - y[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.state_dict()


def save_checkpoint(path, model, opt, sched, step, gen):
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "step": step, "gen": gen.get_state()}, path)


def load_checkpoint(path, model, opt, sched, gen):
    ck = torch.load(path, weights_only=False)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    sched.load_state_dict(ck["sched"])
    gen.set_state(ck["gen"])
    return ck["step"]


def train_steps(model, opt, sched, gen, start, end, data):
    x, y = data
    for _ in range(start, end):
        idx = torch.randint(0, len(x), (32,), generator=gen)
        loss = ((model(x[idx]) - y[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
