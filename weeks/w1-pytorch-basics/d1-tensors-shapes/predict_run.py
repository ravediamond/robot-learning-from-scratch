"""Run AFTER writing your guesses in predict.md."""
import torch

x = torch.zeros(4, 50, 6)

checks = [
    ("1 int list dtype", lambda: torch.tensor([1, 2, 3]).dtype),
    ("2 mixed list dtype", lambda: torch.tensor([1.0, 2, 3]).dtype),
    ("3 mean(dim=1)", lambda: tuple(x.mean(dim=1).shape)),
    ("4 mean(dim=1, keepdim)", lambda: tuple(x.mean(dim=1, keepdim=True).shape)),
    ("5 transpose(1,2)", lambda: tuple(x.transpose(1, 2).shape)),
    ("6 transpose then view", lambda: tuple(x.transpose(1, 2).view(4, -1).shape)),
    ("7 stack vs cat", lambda: (tuple(torch.stack([torch.zeros(6), torch.zeros(6)]).shape),
                                tuple(torch.cat([torch.zeros(6), torch.zeros(6)]).shape))),
    ("8 unfold", lambda: tuple(torch.arange(10).unfold(0, 3, 1).shape)),
    ("9 int / 2", lambda: (torch.tensor([1, 2]) / 2).dtype),
    ("10 fp16 tiny add", lambda: bool((torch.tensor([1.0], dtype=torch.float16)
                                       + torch.tensor([1e-4], dtype=torch.float16))
                                      == torch.tensor([1.0], dtype=torch.float16))),
]

for name, fn in checks:
    try:
        print(f"{name:28s} -> {fn()}")
    except Exception as e:  # some predictions are "it errors"
        print(f"{name:28s} -> ERROR {type(e).__name__}: {str(e)[:90]}")

print("\nWhy 10 matters: fp16 has ~3 decimal digits. Tiny updates vanish. That is why")
print("training uses bf16/fp32 master weights and why you never accumulate in fp16.")
