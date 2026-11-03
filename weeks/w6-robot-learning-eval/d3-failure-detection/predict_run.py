"""Run AFTER writing your guesses AND after your tests pass."""
import math

import torch

import w6d3_monitors as m

lab = torch.tensor([1] * 21 + [0] * 44)
print("1 constant:", m.auroc(torch.zeros(65), lab))
s = torch.cat([torch.randn(21, generator=torch.Generator().manual_seed(0)) - 3, torch.randn(44)])
print("2 reversed:", round(m.auroc(s, lab), 3), "-> use -score (and ask why the direction was wrong)")
g = torch.Generator().manual_seed(1)
s = torch.cat([torch.randn(21, generator=g) + 1.19, torch.randn(44, generator=g)])     # ~0.8 AUROC
print("3 AUROC", round(m.auroc(s, lab), 3), "CI", tuple(round(x, 3) for x in m.auroc_ci(s, lab, 2000, 0)))
for n in (19, 5):
    k = math.ceil((n + 1) * 0.9)
    print(f"4 n={n}: k = {k}" + ("  > n: no finite threshold, flag nothing is the only guarantee" if k > n else ""))
