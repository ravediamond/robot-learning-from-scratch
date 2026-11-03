"""Run AFTER writing your guesses in predict.md."""
import torch

x = torch.arange(12).reshape(3, 4)


def item8():
    a = torch.zeros(3)
    b = a
    b += 1
    return a


def item9():
    a = torch.zeros(3)
    b = a
    b = b + 1
    return a


checks = [
    ("1", lambda: tuple((torch.zeros(3, 1) + torch.zeros(4)).shape)),
    ("2", lambda: tuple((torch.zeros(2, 3) + torch.zeros(3, 2)).shape)),
    ("3", lambda: tuple((torch.zeros(5, 1, 6) - torch.zeros(1, 7, 6)).shape)),
    ("4", lambda: x[[0, 2], [1, 3]]),
    ("5", lambda: tuple(x[:, [1, 3]].shape)),
    ("6", lambda: x[x > 5]),
    ("7", lambda: torch.einsum("ij->j", torch.ones(3, 4))),
    ("8 in-place +=", item8),
    ("9 out-of-place", item9),
    ("10", lambda: torch.ones(2, 3).sum(dim=(0, 1))),
]
for name, fn in checks:
    try:
        print(f"{name:16s} -> {fn()}")
    except Exception as e:
        print(f"{name:16s} -> ERROR {type(e).__name__}: {str(e)[:90]}")

print("\n8 vs 9: in-place ops mutate shared storage. Same trap with tensors returned by a")
print("dataset or a replay buffer: modify them in place and you corrupt the stored data.")
