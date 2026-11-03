"""Run AFTER writing your guesses in predict.md."""
import torch

x = torch.tensor([1.0, 2.0, 3.0])
print("1", torch.softmax(x * 10, 0).round(decimals=4).tolist(), torch.softmax(x * 0.1, 0).round(decimals=3).tolist())
torch.manual_seed(0)
q, k = torch.randn(10000, 256), torch.randn(10000, 256)
dots = (q * k).sum(-1)
print("2 std q.k:", round(dots.std().item(), 2), " /sqrt(d):", round((dots / 16).std().item(), 3))
print("3", torch.tril(torch.ones(3, 3, dtype=torch.bool)).int().tolist())
print("4 position 0 can only see itself: out[0] = v[0]")
print("5", torch.softmax(torch.full((3,), float("-inf")), 0).tolist())
xs = torch.randn(5, 8)
A = torch.softmax(xs @ xs.T, -1)
out = A @ xs
perm = torch.tensor([3, 0, 4, 1, 2])
A2 = torch.softmax(xs[perm] @ xs[perm].T, -1)
print("6 permutation-equivariant:", torch.allclose((A2 @ xs[perm]), out[perm], atol=1e-6))
