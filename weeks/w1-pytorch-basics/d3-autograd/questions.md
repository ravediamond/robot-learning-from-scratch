# Questions (out loud)

1. Derive dW for `pred = x @ W.T + b` with MSE, on the whiteboard, in under 3 minutes. Then dx. Why does dx matter if x is the input data?
2. Why does PyTorch accumulate gradients instead of overwriting them? Give one legitimate use (hint: batches too big for memory).
3. The review bug: describe what the weight actually does over time. Why does it not simply converge more slowly? (Run it and print w every 20 steps if you are unsure.)
4. What is `torch.no_grad()` for, and how is it different from `.detach()`? From `model.eval()`?
5. Stop-gradient in JEPA/BYOL: what happens without it? Why does EMA on the target encoder help on top?
6. Interviewer: "Your loss is NaN after 300 steps." List the first five things you check, in order.
7. Why central differences and float64 for gradient checking? What is the error order of forward vs central differences?
