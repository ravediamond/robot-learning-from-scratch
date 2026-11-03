"""W1 D5: SOLUTION."""
import torch
import torch.nn as nn

Episode = dict[str, torch.Tensor]


def split_by_episode(n_episodes: int, val_frac: float = 0.2, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_episodes, generator=g).tolist()
    n_val = round(val_frac * n_episodes)
    return perm[n_val:], perm[:n_val]


def stack_frames(episodes: list[Episode], ids: list[int]):
    obs = torch.cat([episodes[i]["obs"] for i in ids])       # (N, 12)
    act = torch.cat([episodes[i]["action"] for i in ids])    # (N, 6)
    return obs, act


def compute_norm_stats(obs, act, min_std: float = 1e-2):
    return {
        "obs_mean": obs.mean(0), "obs_std": obs.std(0).clamp(min=min_std),
        "act_mean": act.mean(0), "act_std": act.std(0).clamp(min=min_std),
    }


class BCPolicy(nn.Module):
    def __init__(self, stats: dict[str, torch.Tensor], hidden: int = 256):
        super().__init__()
        for k, v in stats.items():
            self.register_buffer(k, v.clone().float())
        obs_dim, act_dim = stats["obs_mean"].shape[0], stats["act_mean"].shape[0]
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward_normalized(self, obs):
        z = (obs - self.obs_mean) / self.obs_std      # (..., 12)
        return self.net(z)                            # (..., 6), normalized action space

    def forward(self, obs):
        return self.forward_normalized(obs) * self.act_std + self.act_mean   # raw units


def train_bc(obs, act, epochs: int = 40, batch_size: int = 256, lr: float = 1e-3, seed: int = 0):
    torch.manual_seed(seed)
    policy = BCPolicy(compute_norm_stats(obs, act))
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    target = (act - policy.act_mean) / policy.act_std          # (N, 6) normalized
    g = torch.Generator().manual_seed(seed)
    policy.train()
    for _ in range(epochs):
        perm = torch.randperm(len(obs), generator=g)
        for i in range(0, len(obs), batch_size):
            idx = perm[i:i + batch_size]
            loss = ((policy.forward_normalized(obs[idx]) - target[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    policy.eval()
    return policy


def raw_mse(policy, obs, act) -> float:
    with torch.no_grad():
        return ((policy(obs) - act) ** 2).mean().item()


def hold_baseline_mse(obs, act) -> float:
    return ((obs[:, :6] - act) ** 2).mean().item()


def save_policy(policy, path: str, hidden: int = 256) -> None:
    torch.save({"state_dict": policy.state_dict(), "hidden": hidden}, path)


def load_policy(path: str) -> BCPolicy:
    ckpt = torch.load(path, weights_only=True)
    sd = ckpt["state_dict"]
    dummy = {k: torch.zeros_like(sd[k]) for k in ("obs_mean", "obs_std", "act_mean", "act_std")}
    policy = BCPolicy(dummy, hidden=ckpt["hidden"])
    policy.load_state_dict(sd, strict=True)    # raises on any missing/unexpected key
    policy.eval()
    return policy
