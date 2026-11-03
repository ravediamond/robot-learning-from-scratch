# Questions (out loud)

1. Write the five-line training loop on the whiteboard. For each line, say which bug appears if it is missing or misplaced.
2. Overfit-one-batch: what does it prove, and what does it NOT prove? What do you conclude if it fails? If it passes but validation is bad?
3. `model.eval()` vs `torch.no_grad()`: what does each change? Name a layer that behaves differently in eval mode besides Dropout.
4. Why might a robot policy avoid BatchNorm? (Think: batch of 1 at inference on the arm, correlated frames from the same episode in a batch.)
5. Step-0 loss for 10-class classification is 4.7 instead of ~2.3. What does that tell you?
6. Adam vs SGD in one minute: what does Adam keep per parameter, and why is its default lr (1e-3) so different from SGD's (1e-1)?
7. The review policy registers its layers through `add_module`. Is that part a bug? What would break with a plain list and no `add_module`? How would you notice?
8. Interviewer: "Train loss goes down, val loss goes up from epoch 3." Three explanations specific to robot demonstration data, and the check for each.
