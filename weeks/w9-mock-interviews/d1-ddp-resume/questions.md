# Questions (out loud): the "1 GPU to 8 GPUs" round

1. Going from 1 to 8 GPUs, what happens to the effective batch, and what do you do with the learning rate and warmup?
2. Why must every rank call collectives in the same order? Give a real way this hangs (hint: rank-0-only evaluation with a forward pass through the DDP model).
3. BatchNorm under DDP: what goes wrong, and the two fixes. (And why policies avoid BN anyway: W2 D3.)
4. What exactly must a checkpoint contain to resume bit-identically? Which item is most often forgotten?
5. The review bug: why does the loss jump after resume? How long does it take to recover?
6. Interviewer: "Training a 450M VLA on 4 GPUs is 2.5× faster, not 4×. Where did the rest go?" List causes and how you'd measure each.
