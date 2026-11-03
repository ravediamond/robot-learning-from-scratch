"""Run AFTER writing your guesses AND after your tests pass."""
import torch

import w7d1_vicreg as m

data = torch.randn(4000, 2, generator=torch.Generator().manual_seed(0))
for kind in ("invariance", "vicreg", "no_cov"):
    enc, last = m.train_siamese(data, kind)
    with torch.no_grad():
        z = enc(data)
    print(f"{kind:11s} loss {last:.4f}  std {z.std(0).mean():.3f}  effective rank {m.effective_rank(z):.2f}")
print("4 batch of 4: variance from 4 samples is noisy; covariance of 8 dims from 4 samples has rank <= 3")
