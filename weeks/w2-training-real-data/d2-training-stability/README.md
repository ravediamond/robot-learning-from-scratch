# W2 D2: Training that does not blow up

Book: P2.7 (optimisers and schedules), P2.14 (debugging tools). Coding list item 10.

## Four habits

**1. Know the step-0 loss before you see it.** Classification over K classes with a fresh
model: ≈ ln K. Regression with near-zero predictions: ≈ mean(y²) (= 1 with normalized
targets). Far off → bug before any learning (bad init scale, unnormalized targets, wrong labels).

**2. Warm up, then decay.** Adam's second-moment estimate is noisy in the first steps; a large
lr there can wreck the weights. Linear warmup from 0 to the peak over W steps, then cosine
decay to ~0:
```
step < W:   lr = peak * step / W
else:       p = (step - W) / (total - W);  lr = final + (peak - final) * 0.5 * (1 + cos(pi * p))
```
A scheduler is defined in **steps** or in **epochs**. Mixing them is a classic silent bug.

**3. Clip the global gradient norm.** One bad batch (a teleop glitch, a loss spike) can produce
a huge gradient. `g_norm = sqrt(sum of all grad²)`; if `g_norm > max`, scale every grad by
`max / g_norm`. Must happen AFTER `backward()` and BEFORE `step()`. Log `g_norm` every step:
it is the best early-warning signal you have.

**4. Find the first NaN, not the last.** Forward hooks see every module's output. The first
module whose output is non-finite is where to look.
```python
h = module.register_forward_hook(lambda mod, inp, out: ...)
h.remove()
```

## Today's build (`w2d2_stability.py`)
`initial_mse_if_zero`, `initial_ce`, `warmup_cosine_lr`, `clip_grad_norm` (by hand),
`first_nonfinite_module`.

Run: `uv run pytest weeks/w2-training-real-data/d2-training-stability`
