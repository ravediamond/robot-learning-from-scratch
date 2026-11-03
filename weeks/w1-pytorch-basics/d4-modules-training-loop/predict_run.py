"""Run AFTER writing your guesses in predict.md."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


class ListModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(4, 4)]


def i4():
    d = nn.Dropout(0.5)
    d.train()
    return d(torch.ones(8))


def i5():
    d = nn.Dropout(0.5)
    d.eval()
    return d(torch.ones(8))


def i6():
    m = nn.Sequential(nn.Linear(6, 64), nn.ReLU(), nn.Linear(64, 6))
    x, y = torch.randn(1000, 6), torch.randn(1000, 6)
    return round(F.mse_loss(m(x), y).item(), 3)


def i7():
    m = nn.Linear(32, 10)
    x, y = torch.randn(1000, 32), torch.randint(0, 10, (1000,))
    return f"{F.cross_entropy(m(x), y).item():.3f}   (ln 10 = {math.log(10):.3f})"


def i8():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(2, 2)

    m = M()
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    m.fc = nn.Linear(2, 2)
    new_ids = {id(p) for p in m.parameters()}
    opt_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    return f"optimizer holds the new weights: {bool(new_ids & opt_ids)}"


checks = [
    ("1", lambda: n_params(nn.Linear(10, 20))),
    ("2", lambda: n_params(nn.Sequential(nn.Linear(6, 64), nn.ReLU(), nn.Linear(64, 64),
                                         nn.ReLU(), nn.Linear(64, 6)))),
    ("3", lambda: n_params(ListModule())),
    ("4 dropout train", i4), ("5 dropout eval", i5),
    ("6 MSE step 0", i6), ("7 CE step 0", i7), ("8", i8),
]
for name, fn in checks:
    print(f"{name:16s} -> {fn()}")

print("\n4: survivors are scaled by 1/(1-p) so the expected value is unchanged between modes.")
