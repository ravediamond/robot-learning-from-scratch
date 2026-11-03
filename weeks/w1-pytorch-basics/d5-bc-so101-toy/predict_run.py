"""Run AFTER writing your guesses in predict.md."""
import torch

from so101_toy import generate_episodes

eps = generate_episodes(60, seed=0)
obs = torch.cat([e["obs"] for e in eps])        # (N, 12)
act = torch.cat([e["action"] for e in eps])     # (N, 6)
ep_id = torch.cat([torch.full((len(e["obs"]),), i) for i, e in enumerate(eps)])
N = len(obs)
print(f"1 frames: {N}")
std = obs[:, :6].std(dim=0)
print(f"2 per-joint std: {[round(s, 1) for s in std.tolist()]}")
print(f"3 hold-baseline MSE (raw units): {((obs[:, :6] - act) ** 2).mean().item():.2f}")

g = torch.Generator().manual_seed(0)
perm = torch.randperm(N, generator=g)
val_f, train_f = perm[: N // 5], perm[N // 5:]
leak = torch.isin(ep_id[val_f], ep_id[train_f]).float().mean().item()
print(f"4 val frames with an episode-mate in train (frame split): {leak:.0%}")

z = (obs - obs.mean(0)) / obs.std(0).clamp(min=1e-2)   # goal gripper is constant: std 0 -> nan without the clamp


def nn_mse(train_idx, val_idx):
    d = torch.cdist(z[val_idx], z[train_idx])            # (V, Tr)
    nearest = train_idx[d.argmin(dim=1)]
    return ((act[nearest] - act[val_idx]) ** 2).mean().item()


val_ep = torch.isin(ep_id, torch.arange(48, 60))
print(f"5 nearest-neighbor val MSE  frame split: {nn_mse(train_f, val_f):.2f}"
      f"   episode split: {nn_mse(torch.where(~val_ep)[0], torch.where(val_ep)[0]):.2f}")
print("\nThe frame split flatters the dumbest possible policy. That is what leakage looks like.")
