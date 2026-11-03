"""Chunked BC on sim2d demos: split, normalise, train, evaluate, save. Runs without errors. 7 bugs."""
import math

import torch
import torch.nn as nn

import sim2d

H = 8


def split(demos: dict, val_frac: float = 0.2, seed: int = 0) -> tuple[dict, dict]:
    """Train/val split for the demos."""
    g = torch.Generator().manual_seed(seed)
    N = len(demos["obs"])
    perm = torch.randperm(N, generator=g)
    n_val = int(val_frac * N)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    return {k: v[tr_idx] for k, v in demos.items()}, {k: v[val_idx] for k, v in demos.items()}


def norm_stats(demos: dict) -> dict:
    """Per-dim mean/std of observations and actions."""
    acts = demos["chunks"][demos["mask"]]
    return {"obs_mean": demos["obs"].mean(0), "obs_std": demos["obs"].std(0) + 1e-6,
            "act_mean": acts.mean(0), "act_std": acts.std(0) + 1e-6}


class Policy(nn.Module):
    def __init__(self, stats: dict):
        super().__init__()
        self.stats = stats
        self.net = nn.Sequential(nn.Linear(2, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 256), nn.ReLU(),
                                 nn.Linear(256, H * 2))

    def forward(self, obs):
        z = (obs - self.stats["obs_mean"]) / self.stats["obs_std"]
        return self.net(z).view(-1, H, 2)                                  # normalised actions

    def act(self, obs):
        return self(obs) * self.stats["act_std"] + self.stats["act_mean"]


def masked_loss(pred, target, mask):
    err = ((pred - target) ** 2).mean(-1)
    return (err * mask).sum() / err.numel()


def lr_at(step: int, total: int, peak: float = 1e-3, warmup: int = 100) -> float:
    if step < warmup:
        return peak * step / warmup
    return peak * 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total - warmup)))


def train(train_d: dict, stats: dict, steps: int = 2000, batch: int = 256, seed: int = 0) -> Policy:
    torch.manual_seed(seed)
    pol = Policy(stats)
    opt = torch.optim.Adam(pol.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(seed)
    target = (train_d["chunks"] - stats["act_mean"]) / stats["act_std"]
    for step in range(steps):
        for grp in opt.param_groups:
            grp["lr"] = lr_at(step // 100, steps)
        i = torch.randint(0, len(target), (batch,), generator=g)
        loss = masked_loss(pol(train_d["obs"][i]), target[i], train_d["mask"][i].float())
        loss.backward()
        opt.step()
    return pol


@torch.no_grad()
def val_loss(pol: Policy, val_d: dict) -> float:
    target = (val_d["chunks"] - pol.stats["act_mean"]) / pol.stats["act_std"]
    return masked_loss(pol(val_d["obs"]), target, val_d["mask"].float()).item()


def save(pol: Policy, path: str) -> None:
    torch.save(pol.state_dict(), path)


def load(path: str) -> Policy:
    dummy = {"obs_mean": torch.zeros(2), "obs_std": torch.ones(2), "act_mean": torch.zeros(2), "act_std": torch.ones(2)}
    pol = Policy(dummy)
    pol.load_state_dict(torch.load(path, weights_only=True))
    return pol


def main():
    demos = sim2d.make_demos(200, H, seed=0)
    stats = norm_stats(demos)
    tr, va = split(demos)
    pol = train(tr, stats)
    print("val loss", val_loss(pol, va))
    save(pol, "policy.pt")


if __name__ == "__main__":
    main()
