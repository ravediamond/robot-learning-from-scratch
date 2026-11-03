"""Run AFTER writing your guesses in predict.md."""
import math

import torch

H = 50
n = torch.arange(H).float()
C = torch.cos(math.pi * (n[None, :] + 0.5) * n[:, None] / H) * torch.cat([torch.tensor([math.sqrt(1 / H)]), torch.full((H - 1,), math.sqrt(2 / H))])[:, None]
print("1 half a bin:", 2 / 256 / 2)
for name, sig in [("ramp", torch.linspace(-0.5, 0.5, H)), ("step", (n >= 20).float())]:
    e = (C @ sig) ** 2
    k90 = int((torch.cumsum(e, 0) / e.sum() < 0.999).sum()) + 1
    print(f"2/3 {name}: coefficients holding 99.9% of the energy: {k90}")
print("4 binning", H * 6, "tokens; DCT keep 8:", 8 * 6)
