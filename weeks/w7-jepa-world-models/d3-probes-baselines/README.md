# W7 D3: Is my latent any good? Probes and baselines

Primer: `knowledge/05-jepa-ssl.md` ("How you know your latent is any good"). Book: 12.3–12.4, 6.

A world model's loss going down proves nothing. Three cheap instruments, always together:

1. **Linear probe**: freeze the encoder, fit `y ≈ W z + b` for a known quantity y (position, gripper
   opening) on TRAIN episodes, report R² on HELD-OUT episodes. Closed-form ridge:
   `[W; b] = (Xᵀ X + λI)⁻¹ Xᵀ Y` with X = [Z, 1].
2. **MLP probe**: same with a small MLP. Linear low + MLP high = the information is there but
   entangled (nonlinearly encoded). Both low = it is not there.
3. **Copy-last baseline** for prediction: error of "z_{t+1} = z_t". A predictor that does not beat it
   clearly has learned nothing about dynamics (low error can come from slow scenes for free).

`R² = 1 − Σ (y − ŷ)² / Σ (y − ȳ)²` (sums over samples and dims; ȳ from the evaluated set).

## Today's build (`w7d3_probes.py`)
`ridge_probe`, `apply_probe`, `r2`, `mlp_probe`, `copy_last_error`.
Test data: synthetic latents that encode position nonlinearly: linear R² low, MLP R² high.

Run: `uv run pytest weeks/w7-jepa-world-models/d3-probes-baselines`
