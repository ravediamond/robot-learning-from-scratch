"""Run AFTER writing your guesses in predict.md."""
import torch
import torch.nn as nn

print("1", (224 // 16) ** 2, 3 * 16 * 16, " 2", (224 // 14) ** 2, 3 * 14 * 14)
x = torch.randn(8)
th = torch.tensor(0.7)
rot = torch.stack([x[0::2] * th.cos() - x[1::2] * th.sin(), x[0::2] * th.sin() + x[1::2] * th.cos()], -1).flatten()
print("3 norms:", round(x.norm().item(), 4), round(rot.norm().item(), 4))
torch.manual_seed(0)
layer = nn.TransformerEncoderLayer(64, 4, 256, dropout=0.0, batch_first=True, norm_first=True)
h = torch.randn(2, 10, 64)
s0 = h.std().item()
for _ in range(12):
    h = nn.TransformerEncoderLayer(64, 4, 256, dropout=0.0, batch_first=True, norm_first=True)(h)
print(f"4 residual std: input {s0:.2f}  after 12 blocks {h.std().item():.2f}  (each block ADDS to the stream)")
T, D = 50, 16
pos = torch.arange(T)[:, None]
div = 10000 ** (torch.arange(0, D, 2) / D)
pe = torch.zeros(T, D)
pe[:, 0::2], pe[:, 1::2] = torch.sin(pos / div), torch.cos(pos / div)
print("5 PE[t].PE[t+3] for t=0,10,40:", [round((pe[t] @ pe[t + 3]).item(), 4) for t in (0, 10, 40)])
