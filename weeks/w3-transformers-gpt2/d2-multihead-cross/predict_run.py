"""Run AFTER writing your guesses in predict.md."""
import torch
import torch.nn as nn

n = lambda mod: sum(p.numel() for p in mod.parameters())
print("1", n(nn.MultiheadAttention(64, 8)), " 2", n(nn.MultiheadAttention(64, 1)))
x = torch.arange(24).view(1, 2, 12)
print("3 right split, head 0:", x.view(1, 2, 3, 4).transpose(1, 2)[0, 0].tolist())
print("4 wrong split, 'head' 0:", x.view(1, 3, 2, 4)[0, 0].tolist(), " <- mixes the two time steps")
print("5 per head (50, 300)")
