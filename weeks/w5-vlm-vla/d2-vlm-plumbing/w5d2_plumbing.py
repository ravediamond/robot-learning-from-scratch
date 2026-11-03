"""W5 D2: SOLUTION."""
import torch
import torch.nn as nn


def pixel_shuffle_tokens(x, r):
    B, N, C = x.shape
    h = int(round(N ** 0.5))
    g = x.view(B, h // r, r, h // r, r, C)                    # (B, i, ri, j, rj, C)
    g = g.permute(0, 1, 3, 2, 4, 5)                           # (B, i, j, ri, rj, C)
    return g.reshape(B, (h // r) ** 2, r * r * C)


class Projector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))

    def forward(self, x):
        return self.net(x)


def block_attention_mask(block_sizes, causal_last=False):
    ids = torch.cat([torch.full((n,), b) for b, n in enumerate(block_sizes)])    # block id per token
    mask = ids[None, :] <= ids[:, None]                                         # key block <= query block
    if causal_last:
        n = block_sizes[-1]
        mask[-n:, -n:] = torch.tril(torch.ones(n, n, dtype=torch.bool))
    return mask


def assemble_prefix(img_tokens, txt_tokens, txt_mask, state_token):
    B = txt_tokens.shape[0]
    seq = torch.cat([*img_tokens, txt_tokens, state_token], dim=1)
    masks = [torch.ones(B, t.shape[1], dtype=torch.bool) for t in img_tokens]
    km = torch.cat([*masks, txt_mask, torch.ones(B, 1, dtype=torch.bool)], dim=1)
    return seq, km
