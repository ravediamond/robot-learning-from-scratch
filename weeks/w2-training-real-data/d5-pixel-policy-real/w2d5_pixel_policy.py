"""W2 D5: SOLUTION."""
import torch
import torch.nn as nn

from cnn_ref import CNNEncoder, random_shift


def masked_chunk_mse(pred, target, mask):
    err = ((pred - target) ** 2).mean(-1)                    # (N, H)
    m = mask.to(err.dtype)
    return (err * m).sum() / m.sum().clamp(min=1.0)


class PixelStatePolicy(nn.Module):
    def __init__(self, stats, H, use_image=True, hidden=256):
        super().__init__()
        for k, v in stats.items():
            self.register_buffer(k, v.clone().float())
        Ds, Da = stats["state_mean"].shape[0], stats["act_mean"].shape[0]
        self.H, self.Da, self.use_image = H, Da, use_image
        self.cnn = CNNEncoder(64) if use_image else None
        self.state_fc = nn.Linear(Ds, 64)
        self.head = nn.Sequential(nn.Linear(128 if use_image else 64, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, H * Da))

    def forward_normalized(self, img, state):
        s = self.state_fc((state - self.state_mean) / self.state_std)          # (N, 64)
        feat = torch.cat([self.cnn(img), s], -1) if self.use_image else s
        return self.head(torch.relu(feat)).view(-1, self.H, self.Da)

    def forward(self, img, state):
        return self.forward_normalized(img, state) * self.act_std + self.act_mean


def _stats(data):
    acts = data["actions"][data["mask"]]                                        # (real steps, Da)
    return {"state_mean": data["state"].mean(0), "state_std": data["state"].std(0).clamp(min=1e-3),
            "act_mean": acts.mean(0), "act_std": acts.std(0).clamp(min=1e-3)}


def train_policy(data, use_image, epochs=20, batch_size=64, lr=1e-3, aug_pad=2, seed=0):
    torch.manual_seed(seed)
    pol = PixelStatePolicy(_stats(data), H=data["actions"].shape[1], use_image=use_image)
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    N = len(data["state"])
    target = (data["actions"] - pol.act_mean) / pol.act_std
    pol.train()
    for _ in range(epochs):
        perm = torch.randperm(N, generator=g)
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            img = random_shift(data["img"][idx], aug_pad, g) if use_image else data["img"][idx]
            loss = masked_chunk_mse(pol.forward_normalized(img, data["state"][idx]), target[idx], data["mask"][idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    pol.eval()
    return pol


def evaluate(policy, data):
    tot, n = 0.0, 0.0
    with torch.no_grad():
        for i in range(0, len(data["state"]), 512):
            sl = slice(i, i + 512)
            pred = policy(data["img"][sl], data["state"][sl])
            err = ((pred - data["actions"][sl]) ** 2).mean(-1)
            m = data["mask"][sl].float()
            tot += (err * m).sum().item()
            n += m.sum().item()
    return tot / n
