"""Run AFTER writing your guesses in predict.md."""
import math

import torch

a = torch.tensor([1.0] * 70 + [-1.0] * 30)
print("1 mean:", a.mean().item(), " median:", a.median().item())
print("2 a sampler outputs +1 70% of the time and -1 30%: never 0.4")
kl = lambda mu, logvar: 0.5 * (math.exp(logvar) + mu ** 2 - 1 - logvar)
print("3", kl(0, 0), kl(2, 0))
print("4 std:", math.exp(0.5 * 0), round(math.exp(0.5 * 2), 3))
print("5 huge beta: q(z|x) = N(0, I) for every x -> z carries no information -> decoder outputs the MEAN: MSE again")
