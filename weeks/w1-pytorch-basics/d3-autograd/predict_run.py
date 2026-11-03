"""Run AFTER writing your guesses in predict.md."""
import warnings

import torch

warnings.filterwarnings("ignore")


def i1():
    x = torch.tensor(3.0, requires_grad=True)
    (x ** 2).backward()
    return x.grad


def i2():
    x = torch.tensor(3.0, requires_grad=True)
    (x ** 2).backward()
    (x ** 2).backward()
    return x.grad


def i3():
    x = torch.tensor(3.0, requires_grad=True)
    y = x ** 2
    y.backward()
    y.backward()
    return x.grad


def i4():
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    (x * 2).backward()
    return x.grad


def i5():
    x = torch.tensor(2.0, requires_grad=True)
    z = (x * 3).detach() * x
    z.backward()
    return x.grad


def i6():
    return torch.tensor(2, requires_grad=True)


def i7():
    a = torch.tensor(1.0, requires_grad=True)
    b = a * 2
    b.backward()
    return dict(b_is_leaf=b.is_leaf, a_is_leaf=a.is_leaf, b_grad=b.grad, a_grad=a.grad)


def i8():
    x = torch.tensor(1.0, requires_grad=True)
    with torch.no_grad():
        y = x * 2
    return y.requires_grad


def i9():
    x = torch.tensor(1.0, requires_grad=True)
    torch.relu(x - 2).backward()
    return x.grad


for name, fn in [("1", i1), ("2", i2), ("3", i3), ("4", i4), ("5", i5), ("6", i6),
                 ("7", i7), ("8", i8), ("9", i9)]:
    try:
        print(f"{name:3s} -> {fn()}")
    except Exception as e:
        print(f"{name:3s} -> ERROR {type(e).__name__}: {str(e)[:100]}")

print("\n9 is a 'dead ReLU': zero gradient, the weight never moves. If a whole layer is")
print("dead at init (bad bias, bad LR), the network stops learning silently.")
