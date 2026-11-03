"""W5 D1: SOLUTION."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def clip_loss(img_emb, txt_emb, logit_scale):
    z, u = F.normalize(img_emb, dim=-1), F.normalize(txt_emb, dim=-1)
    logits = logit_scale.exp() * z @ u.T                                  # (N, N)
    labels = torch.arange(len(z), device=z.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def siglip_loss(img_emb, txt_emb, log_t, b):
    z, u = F.normalize(img_emb, dim=-1), F.normalize(txt_emb, dim=-1)
    logits = log_t.exp() * z @ u.T + b
    labels = 2 * torch.eye(len(z), device=z.device) - 1
    return -F.logsigmoid(labels * logits).mean()


class TinyCLIP(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.img = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                 nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(2),
                                 nn.Flatten(), nn.Linear(256, d))
        self.tok = nn.Embedding(9, d)
        self.txt = nn.Linear(d, d)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def encode_image(self, img):
        return self.img(img)

    def encode_text(self, tokens):
        return self.txt(self.tok(tokens).sum(1))


def train_clip(imgs, caps, steps=1500, batch=64, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    model = TinyCLIP()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        i = torch.randint(0, len(imgs), (batch,), generator=g)
        loss = clip_loss(model.encode_image(imgs[i]), model.encode_text(caps[i]), model.logit_scale)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.eval()


@torch.no_grad()
def zero_shot_accuracy(model, imgs, caps, all_caps):
    z = F.normalize(model.encode_image(imgs), dim=-1)
    u = F.normalize(model.encode_text(all_caps), dim=-1)
    pred = all_caps[(z @ u.T).argmax(1)]                                  # (N, 3)
    return (pred == caps).all(1).float().mean().item()
