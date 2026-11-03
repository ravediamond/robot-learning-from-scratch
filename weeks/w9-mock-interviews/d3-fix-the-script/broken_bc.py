"""SOLUTION: the 7 fixes are marked FIX."""
import math

import torch
import torch.nn as nn

import sim2d

H = 8


def split(demos: dict, val_frac: float = 0.2, seed: int = 0) -> tuple[dict, dict]:
    """Train/val split for the demos."""
    g = torch.Generator().manual_seed(seed)
    eps = torch.unique(demos["episode"])                               # FIX 1: split by EPISODE, not by frame
    perm = eps[torch.randperm(len(eps), generator=g)]
    val_eps = perm[: max(1, int(val_frac * len(eps)))]
    is_val = torch.isin(demos["episode"], val_eps)
    return {k: v[~is_val] for k, v in demos.items()}, {k: v[is_val] for k, v in demos.items()}


def norm_stats(demos: dict) -> dict:
    """Per-dim mean/std of observations and actions."""
    acts = demos["chunks"][demos["mask"]]
    return {"obs_mean": demos["obs"].mean(0), "obs_std": demos["obs"].std(0) + 1e-6,
            "act_mean": acts.mean(0), "act_std": acts.std(0) + 1e-6}


class Policy(nn.Module):
    def __init__(self, stats: dict):
        super().__init__()
        for k, v in stats.items():                                      # FIX 6: stats as buffers (saved/loaded)
            self.register_buffer(k, v.clone())
        self.net = nn.Sequential(nn.Linear(2, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 256), nn.ReLU(),
                                 nn.Linear(256, H * 2))

    def forward(self, obs):
        z = (obs - self.obs_mean) / self.obs_std
        return self.net(z).view(-1, H, 2)                                  # normalised actions

    def act(self, obs):
        return self(obs) * self.act_std + self.act_mean


def masked_loss(pred, target, mask):
    err = ((pred - target) ** 2).mean(-1)
    return (err * mask).sum() / mask.sum().clamp(min=1.0)                # FIX 3: divide by real steps


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
            grp["lr"] = lr_at(step, steps)                              # FIX 4: schedule in steps, not step // 100
        i = torch.randint(0, len(target), (batch,), generator=g)
        loss = masked_loss(pol(train_d["obs"][i]), target[i], train_d["mask"][i].float())
        opt.zero_grad()                                                 # FIX 7: clear gradients every step
        loss.backward()
        opt.step()
    return pol.eval()


@torch.no_grad()
def val_loss(pol: Policy, val_d: dict) -> float:
    was = pol.training
    pol.eval()                                                          # FIX 5: dropout off during validation
    target = (val_d["chunks"] - pol.act_mean) / pol.act_std
    out = masked_loss(pol(val_d["obs"]), target, val_d["mask"].float()).item()
    pol.train(was)
    return out


def save(pol: Policy, path: str) -> None:
    torch.save(pol.state_dict(), path)


def load(path: str) -> Policy:
    dummy = {"obs_mean": torch.zeros(2), "obs_std": torch.ones(2), "act_mean": torch.zeros(2), "act_std": torch.ones(2)}
    pol = Policy(dummy)
    pol.load_state_dict(torch.load(path, weights_only=True))
    return pol


def main():
    demos = sim2d.make_demos(200, H, seed=0)
    tr, va = split(demos)
    stats = norm_stats(tr)                                              # FIX 2: stats from the training split only
    pol = train(tr, stats)
    print("val loss", val_loss(pol, va))
    save(pol, "policy.pt")


if __name__ == "__main__":
    main()
