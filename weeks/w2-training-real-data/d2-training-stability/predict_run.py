"""Run AFTER writing your guesses in predict.md."""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
x = torch.randn(512, 64)
print("1 CE at init:", round(F.cross_entropy(nn.Linear(64, 100)(x), torch.randint(0, 100, (512,))).item(), 3))
y = 50 + torch.randn(512, 1)
print("2 MSE at init:", round(F.mse_loss(nn.Linear(64, 1)(x), y).item(), 1))
a, b = torch.zeros(2, requires_grad=True), torch.zeros(2, requires_grad=True)
a.grad, b.grad = torch.tensor([3.0, 4.0]), torch.tensor([0.0, 12.0])
ret = torch.nn.utils.clip_grad_norm_([a, b], 1.0)
print("3/4 returned:", ret.item(), " grad a after clip:", a.grad.tolist())
print("5 exp(100):", torch.exp(torch.tensor(100.0)).item(), torch.exp(torch.tensor(100.0, dtype=torch.float64)).item())
print("6 log(softmax):", torch.log(torch.softmax(torch.tensor([0.0, 200.0]), 0)).tolist(),
      " log_softmax:", torch.log_softmax(torch.tensor([0.0, 200.0]), 0).tolist())
