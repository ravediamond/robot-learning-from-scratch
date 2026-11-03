"""Run AFTER writing your guesses AND after your tests pass."""
import torch

import w4d4_cfg as m
from toy2d import CENTRES, clusters

x, y = clusters(4000, torch.Generator().manual_seed(0))
model = m.train_cond_fm(x, y)
req = torch.arange(4).repeat(250)
for w in (0.0, 1.0, 4.0):
    s = m.cfg_sample(model, req, 10, w, torch.Generator().manual_seed(1))
    hit = (torch.cdist(s, CENTRES).argmin(1) == req).float().mean().item()
    spread = (s - CENTRES[req]).norm(dim=-1).mean().item()
    print(f"w={w}: in requested corner {hit:.2f}   mean distance to its centre {spread:.3f}")
print("4 with p_uncond=0 the null embedding is never trained: v_uncond is garbage, CFG extrapolates from garbage")
