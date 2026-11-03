# W1 D3: Autograd

Book: P1 (chain rule, `00p1-deep-learning-from-zero.md`), P2.3, P2.19 (a full autograd engine).
Coding list item 08 (linear forward + manual backward).

## The one idea

Training = "move each weight a little in the direction that lowers the loss". That
direction is the gradient. Autograd computes it for you by recording every operation in
the forward pass (a graph) and applying the chain rule backwards.

You must be able to do it by hand for a linear layer, because interviews ask, and because
it is how you check autograd is computing what you think.

## Linear layer + MSE, by hand

```
pred = x @ W.T + b          x: (N, I)  W: (O, I)  b: (O,)  pred: (N, O)
L    = mean((pred - y)^2)   over all N*O entries

g    = dL/dpred = 2 (pred - y) / (N*O)      (N, O)
dW   = g.T @ x                               (O, I)   same shape as W, always
db   = g.sum(dim=0)                          (O,)
dx   = g @ W                                 (N, I)
```
Trick to never get it wrong: **the gradient has the shape of the thing**. Find the one
matmul arrangement that produces that shape.

## The autograd API

```python
w = torch.zeros(3, requires_grad=True)   # a leaf we want gradients for
loss = f(w)                              # forward builds the graph
loss.backward()                          # fills w.grad (ACCUMULATES: adds to what's there)
with torch.no_grad():                    # updates must not be recorded in the graph
    w -= lr * w.grad
w.grad.zero_()                           # or optimizer.zero_grad()
y.detach()                               # same values, cut from the graph: no gradient flows into it
```

`detach` is the "stop-gradient" of JEPA and BYOL: the target branch gives a value to match
but must not be pushed toward the prediction. Remove it and both sides can collapse to a
constant. You will meet this again in week 7.

## Finite differences: the universal checker

`df/dx_i ≈ (f(x + eps e_i) − f(x − eps e_i)) / (2 eps)`. Slow but always right. Use float64.

## Today's build (`w1d3_autograd.py`)

1. `linear_mse_grads`: dW, db by hand. Test compares to autograd.
2. `finite_diff_grad`: central differences.
3. `fit_line`: gradient descent with autograd, no optimizer class.
4. `stop_gradient_mse`: an MSE where only `pred` gets a gradient.

Run: `uv run pytest weeks/w1-pytorch-basics/d3-autograd`
