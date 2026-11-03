# W4 D2: Diffusion (DDPM) from scratch, in 2-D

Primer: `knowledge/04-diffusion-flow-matching.md`. Diffusion Policy uses exactly this, with
action chunks instead of 2-D points.

```
betas: linear from 1e-4 to 0.02 over T steps;  alpha_t = 1 - beta_t;  abar_t = prod_{s<=t} alpha_s
forward (closed form):  x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps          eps ~ N(0, I)
train:   t ~ U{0..T-1},  loss = || eps - eps_theta(x_t, t) ||^2
sample:  x ~ N(0, I);  for t = T-1 .. 0:
             x = (x - beta_t / sqrt(1 - abar_t) * eps_theta(x, t)) / sqrt(alpha_t)
             if t > 0: x += sqrt(beta_t) * z
```
Indices are 0-based here: `abar[0]` is after one noising step. The network gets t as features
(`toy2d.time_embedding(t / T)`), concatenated with x.

## Today's build (`w4d2_ddpm.py`)
`linear_betas`, `alphas_bar`, `q_sample`, `Denoiser`, `ddpm_loss`, `sample`, `train_ddpm`.
Test: samples land on the two moons (mean nearest-data distance < 0.07).

Run: `uv run pytest weeks/w4-generative-flow/d2-ddpm`
