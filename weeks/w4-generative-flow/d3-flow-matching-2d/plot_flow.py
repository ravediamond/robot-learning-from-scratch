"""PROVIDED (optional): plot data vs 1-step vs 10-step samples with YOUR code."""
import matplotlib.pyplot as plt
import torch

import w4d3_flow as m
from toy2d import two_moons

data = two_moons(4000, torch.Generator().manual_seed(0))
model = m.train_fm(data)
fig, ax = plt.subplots(1, 4, figsize=(16, 4))
for a, (title, pts) in zip(ax, [("data", data[:1000]),
                                ("1 step", m.euler_sample(model, 1000, 1, torch.Generator().manual_seed(1))),
                                ("3 steps", m.euler_sample(model, 1000, 3, torch.Generator().manual_seed(1))),
                                ("10 steps", m.euler_sample(model, 1000, 10, torch.Generator().manual_seed(1)))]):
    a.scatter(pts[:, 0], pts[:, 1], s=2)
    a.set_title(title)
    a.set_xlim(-1.5, 1.5)
    a.set_ylim(-1.2, 1.2)
fig.savefig("flow_samples.png", dpi=120)
print("saved flow_samples.png")
