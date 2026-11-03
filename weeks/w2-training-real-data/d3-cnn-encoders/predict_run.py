"""Run AFTER writing your guesses in predict.md."""
import torch
import torch.nn as nn
import torch.nn.functional as F

print("1", tuple(nn.Conv2d(3, 16, 5, stride=2, padding=2)(torch.zeros(1, 3, 96, 96)).shape))
print("2", tuple(F.unfold(torch.zeros(1, 3, 8, 8), kernel_size=3).shape))


def rf(layers):
    r, j = 1, 1
    for k, s in layers:
        r += (k - 1) * j
        j *= s
    return r


print("3", rf([(3, 1)] * 3), rf([(3, 2)] * 3))
try:
    print("4", nn.BatchNorm1d(4).train()(torch.randn(1, 4)))
except Exception as e:
    print("4 ERROR", type(e).__name__, str(e)[:80])
x = torch.arange(480 * 640 * 3).reshape(480, 640, 3)
print("5 same image:", torch.equal(x.view(3, 480, 640), x.permute(2, 0, 1)))
print("6 constant map -> softmax is uniform -> expected position is the image centre (0, 0)")
