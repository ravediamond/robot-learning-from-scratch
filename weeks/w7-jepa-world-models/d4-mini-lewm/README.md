# W7 D4: A JEPA world model from pixels: mini LeWM (capstone, part 1)

Book: 12a.2 (LeWorldModel), 12.3–12.4 (your W1/W2 models). Primers 05, 06.
World: `pixel_world.py` (32×32 images of a blob that moves with the action). Tools: `jepa_ref.py`
(SIGReg), `probes_ref.py` (probes, copy-last).

## The model (LeWM's recipe, tiny)
```
encoder:   image (1,32,32) -> Conv(1,16,4,s2,p1) ReLU Conv(16,32,4,s2,p1) ReLU Conv(32,32,4,s2,p1) ReLU
           -> Flatten (512) -> Linear(512,128) ReLU -> Linear(128, D=8)
predictor: z_{t+1}_hat = z_t + MLP([z_t, a_t / AMAX])           (residual; MLP 10 -> 128 -> 128 -> 8)
loss:      MSE(z_{t+1}_hat, enc(o_{t+1})) + λ · SIGReg(cat(z_t, z_{t+1}))
           no EMA, no stop-gradient: both sides are the same encoder, trained end to end
```

## The experiment, in this order
1. λ = 0. Watch the collapse (z std → 0, prediction loss → 0).
2. λ = 0.1. Measure: z std, prediction error vs copy-last on held-out episodes, linear AND MLP probe R²
   for the position on held-out episodes.
Write the numbers in PROGRESS.md and compare with your W1 numbers in the register.

## Today's build (`w7d4_lewm.py`)
`Encoder`, `Predictor`, `lewm_loss`, `train_lewm`, `latent_report`.
Tests (~1.5 min): λ = 0 collapses; λ = 0.1 beats copy-last by 3× and MLP probe R² > 0.9.

Run: `uv run pytest weeks/w7-jepa-world-models/d4-mini-lewm -s`
