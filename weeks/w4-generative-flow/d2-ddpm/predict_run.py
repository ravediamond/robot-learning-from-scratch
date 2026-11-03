"""Run AFTER writing your guesses in predict.md."""
import torch

for T in (100, 1000):
    abar = torch.cumprod(1 - torch.linspace(1e-4, 0.02, T), 0)
    print(f"1/2 T={T}: abar[-1] = {abar[-1].item():.4f}  signal std {abar[-1].sqrt().item():.3f}  noise std {(1 - abar[-1]).sqrt().item():.3f}")
print("3 predicting 0 for eps ~ N(0, I) in 2-D: E||eps||^2 = 2 (per-dim mean 1)")
print("4 T network calls per sample: why DDIM (10-16 steps) and flow matching (~10) matter on edge hardware")
