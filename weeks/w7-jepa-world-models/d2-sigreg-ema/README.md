# W7 D2: SIGReg (LeWM's regulariser), EMA targets and stop-gradient

Primer: `knowledge/05-jepa-ssl.md`. Book: 12a.2 (LeWorldModel), 12.2. Coding list items 14, 15.

## SIGReg: make the embedding look like N(0, I), one direction at a time
If z ~ N(0, I) in d dims, every projection u·z on a unit vector u is N(0, 1). So: draw M random unit
directions, project the batch, and test each 1-D sample against N(0, 1). LeJEPA/LeWM use the
**Epps–Pulley** statistic, built on the characteristic function φ(t) = E[exp(i t x)]:
```
p = z @ U                     (N, M) projections, U (d, M) unit columns
t = linspace(-3, 3, 17)
re = mean_n cos(t p_n),  im = mean_n sin(t p_n)                (M, K)
EP = N * ∫ [ (re − e^{−t²/2})² + im² ] e^{−t²/2} dt            trapezoid over t, mean over M
```
Collapsed z → p constant → re = cos(t c) ≠ e^{−t²/2}: big penalty. A 1-D test needs few samples,
so it scales to large d with modest batches. One weight to tune (λ).

## EMA target and stop-gradient (BYOL, I-JEPA, V-JEPA)
The other family: the target encoder is an exponential moving average of the online encoder,
`θ_target ← m θ_target + (1 − m) θ_online` (m ≈ 0.99–0.999), and receives NO gradient
(`.detach()` / `torch.no_grad()`). LeWM drops both and uses SIGReg instead.

## Today's build (`w7d2_sigreg.py`)
`epps_pulley`, `sigreg`, `ema_update`, `jepa_ema_loss`.

Run: `uv run pytest weeks/w7-jepa-world-models/d2-sigreg-ema`
