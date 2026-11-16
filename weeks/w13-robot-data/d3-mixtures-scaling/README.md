# W13 D3: mixtures, sampling weights, scaling, ablations

> **Full lesson for this day:** [`../README.md`](../README.md#day-3-mixtures-scaling-ablations) (the week page: concepts from zero, examples, traps).
> This file is the one-page summary. Order of work: `predict.md` → the lesson → `predict_run.py` → the exercise → the review file → `questions.md`.

Primer: `knowledge/14-robot-data-engineering.md` §3.

## The one idea

A modern robot policy trains on a **mixture**: your 50 hours, a public 1,000-hour dataset, sim data,
other embodiments. The mixture weights matter as much as the architecture.

- **Temperature sampling.** Sample source i with probability ∝ `n_i^(1/T)`. T = 1: proportional to size
  (the big source drowns yours). T → ∞: uniform over sources. Values around 2–5 are common.
- **Per-item weights.** A sampler draws items, not sources: item weight = `p_source / n_items_in_source`.
- **Epochs per source.** With a budget of B samples, source i is seen `B·p_i / n_i` times: a small source
  with a big weight is repeated (overfitting risk).
- **Scaling.** Fit `error ≈ a · N^(-b)` on a log-log plot from a few dataset sizes; it tells you what more
  data of THIS kind buys. Extrapolate a little, never a lot.
- **Ablations with seeds.** "Remove source X" changes success by −4 points; seed-to-seed noise is ±6.
  Pair runs by seed and report the effect with an interval, or you are reading noise.

## Today's build (`w13d3_mixtures.py`)

`temperature_weights`, `item_weights`, `epochs_per_source`, `fit_power_law`, `ablation_effect`.

## Staff angle

Mixture questions are where data engineers and researchers meet. "Here is the scaling curve of our
own data, the marginal value of the public dataset per hour, and the seed noise of the measurement" is
how you turn a data budget into a decision.
