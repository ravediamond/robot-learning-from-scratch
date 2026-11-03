"""Run AFTER writing your guesses AND after your tests pass."""
import torch

import w4d3_flow as m
from toy2d import mean_nn_distance, two_moons

data = two_moons(4000, torch.Generator().manual_seed(0))
model = m.train_fm(data)
for steps in (1, 3, 10, 50):
    s = m.euler_sample(model, 1000, steps, torch.Generator().manual_seed(1))
    print(f"{steps:2d} steps: nn-dist {mean_nn_distance(s, data):.3f}  std {s.std(0).tolist()}  mean {s.mean(0).tolist()}")
print("data std", data.std(0).tolist(), " mean", data.mean(0).tolist())
s10 = m.euler_sample(model, 500, 10, torch.Generator().manual_seed(3))
s50 = m.euler_sample(model, 500, 50, torch.Generator().manual_seed(3))
print("3 same noise, 10 vs 50 steps: mean |diff|", round((s10 - s50).norm(dim=-1).mean().item(), 3))
g = torch.Generator().manual_seed(5)
print("4 final loss:", round(sum(m.fm_loss(model, data[:1000], g).item() for _ in range(20)) / 20, 3),
      "-> not 0: the target x1-x0 is random given (x_t, t); the net learns its conditional mean")
