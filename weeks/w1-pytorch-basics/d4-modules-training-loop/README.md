# W1 D4: nn.Module and the training loop

Book: P2.4 to P2.9, P2.14. Coding list items 03 (training loop), 09 (tiny MLP).

## The one idea

Every training script in the world is the same five lines. Learn them so well you could
write them asleep, and know what each line protects against.

```python
model.train()
for xb, yb in train_loader:
    pred = model(xb)                 # forward
    loss = loss_fn(pred, yb)
    opt.zero_grad()                  # 1. clear old grads (yesterday's bug)
    loss.backward()                  # 2. compute new grads
    opt.step()                       # 3. update weights

model.eval()                         # dropout off, BatchNorm frozen
with torch.no_grad():                # no graph: less memory, faster
    val = mean(loss_fn(model(xb), yb) for xb, yb in val_loader)
```

## nn.Module

```python
class MLP(nn.Module):
    def __init__(self, ...):
        super().__init__()                    # forget this and nothing registers
        self.net = nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o))
    def forward(self, x):
        return self.net(x)
```
Anything assigned as `self.something = nn.Linear(...)` is registered: it appears in
`model.parameters()` and in `model.state_dict()`. A plain Python list of layers is NOT
registered: use `nn.ModuleList`. (Classic silent bug: the optimizer never sees those weights.)

## The overfit-one-batch test

Before training on the full data, train on ONE batch for a few hundred steps. Loss must go
to ~0. If it does not, the bug is in the code (shapes, loss, optimizer wiring), not in the
data or the method. This is the single most useful debugging habit, and interviewers ask for it.

## Loss at step 0

Know what the first loss should be before you see it. Regression with targets normalized to
N(0, 1) and a small init: MSE ≈ 1. Classification with K classes: cross-entropy ≈ ln(K).
If step-0 loss is far off, something is wrong before any learning happens.

## Today's build (`w1d4_training.py`)

1. `MLP`: configurable depth, `nn.ModuleList` or `nn.Sequential`.
2. `count_params`.
3. `train_epoch` and `evaluate`.
4. `overfit_one_batch`.

Run: `uv run pytest weeks/w1-pytorch-basics/d4-modules-training-loop`
