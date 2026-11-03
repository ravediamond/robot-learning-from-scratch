"""Run AFTER writing your guesses in predict.md."""
import math

import torch

import shapes_data

for N in (64, 1024):
    print(f"1 N={N}: ln N = {math.log(N):.2f} (uniform over the batch)")
_, caps = shapes_data.make_shapes(64, torch.Generator().manual_seed(0))
ids = shapes_data.caption_id(caps)
dup = (torch.bincount(ids, minlength=24)[ids] > 1).sum().item()
print(f"2 images sharing their caption with another one in a batch of 64: {dup}")
print("3 no: identical captions have identical text embeddings, so the softmax splits mass between duplicates")
print(f"4 SigLIP at logits 0: -logsigmoid(0) = ln 2 = {math.log(2):.3f}")
