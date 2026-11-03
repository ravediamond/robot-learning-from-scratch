"""W5 D5: SOLUTION."""
import torch
import torch.nn as nn

from vla_blocks import Block, time_embedding

H_DEFAULT = 4


class MiniVLA(nn.Module):
    def __init__(self, D=64, H=H_DEFAULT, heads=4, patch=4, img_size=16, vocab=6):
        super().__init__()
        self.H, self.patch = H, patch
        n_patch = (img_size // patch) ** 2
        self.patch_proj = nn.Linear(3 * patch * patch, D)
        g = img_size // patch
        centres = (torch.arange(g, dtype=torch.float32) + 0.5) / g * 2 - 1           # patch centres in [-1, 1]
        yy, xx = torch.meshgrid(centres, centres, indexing="ij")
        self.register_buffer("patch_xy", torch.stack([xx.flatten(), yy.flatten()], -1))   # (n_patch, 2)
        self.img_pos = nn.Linear(2, D)                                                  # position from coordinates
        self.tok = nn.Embedding(vocab, D)
        self.txt_pos = nn.Parameter(0.02 * torch.randn(4, D))
        self.state_proj = nn.Linear(2, D)
        self.prefix = nn.ModuleList([Block(D, heads), Block(D, heads)])
        self.prefix_ln = nn.LayerNorm(D)
        self.act_in = nn.Linear(2, D)
        self.act_pos = nn.Parameter(0.02 * torch.randn(H, D))
        self.time_proj = nn.Linear(32, D)
        self.cross = nn.ModuleList([Block(D, heads)])
        self.selfb = nn.ModuleList([Block(D, heads)])
        self.out = nn.Linear(D, 2)
        self.register_buffer("causal", torch.tril(torch.ones(H, H, dtype=torch.bool)))

    def patchify(self, img):
        B, C, Hh, W = img.shape
        p = self.patch
        x = img.view(B, C, Hh // p, p, W // p, p).permute(0, 2, 4, 1, 3, 5)
        return x.reshape(B, (Hh // p) * (W // p), C * p * p)

    def encode_prefix(self, img, tokens, state):
        x = torch.cat([self.patch_proj(self.patchify(img)) + self.img_pos(self.patch_xy),
                       self.tok(tokens) + self.txt_pos,
                       self.state_proj(state)[:, None]], dim=1)          # (B, 16 + 4 + 1, D)
        for blk in self.prefix:
            x = blk(x)
        return self.prefix_ln(x)

    def expert(self, prefix, x_t, t):
        a = self.act_in(x_t) + self.act_pos + self.time_proj(time_embedding(t, 32))[:, None]
        for cb, sb in zip(self.cross, self.selfb):
            a = cb(a, context=prefix)
            a = sb(a, mask=self.causal)
        return self.out(a)

    def forward(self, img, tokens, state, x_t, t):
        return self.expert(self.encode_prefix(img, tokens, state), x_t, t)

    @torch.no_grad()
    def act(self, img, tokens, state, generator=None, steps=10):
        prefix = self.encode_prefix(img, tokens, state)                  # once per inference
        x = torch.randn(len(img), self.H, 2, generator=generator)
        for i in range(steps):
            x = x + self.expert(prefix, x, torch.full((len(img),), i / steps)) / steps
        return x


def train_vla(demos, steps=3000, batch=128, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    model = MiniVLA(H=demos["chunk"].shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    g = torch.Generator().manual_seed(seed)
    N = len(demos["state"])
    for step in range(steps):
        for grp in opt.param_groups:
            grp["lr"] = lr * min(1.0, (step + 1) / 200)
        i = torch.randint(0, N, (batch,), generator=g)
        x1 = demos["chunk"][i]
        x0 = torch.randn(x1.shape, generator=g)
        t = torch.rand(batch, generator=g)
        xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
        v = model(demos["img"][i], demos["tokens"][i], demos["state"][i], xt, t)
        loss = ((v - (x1 - x0)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model.eval()
