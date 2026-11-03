# W6 D1: ACT: a conditional VAE over action chunks, plus temporal ensembling

Book: 11.2 (chunking). Primer: `knowledge/03-vla.md`. ACT (Zhao et al. 2023) is the classic
small-data policy (~50 demos) and a LeRobot default for the SO-101.

## The model (MLP version of the transformer original)
```
training:  encoder(obs, true chunk) -> mu, logvar;  z = mu + std * eps
           decoder(obs, z) -> predicted chunk
           loss = masked L1(pred, chunk) + beta * KL(q(z|obs,chunk) || N(0, I))
test time: z = 0 (the prior mean) -> decoder(obs, 0)
```
z is meant to capture the "style" of a demo that the observation alone cannot explain.
Note the loss: **L1**, not MSE. L1's optimum is a median, not a mean: on bimodal data it tends to
pick a side rather than average. Keep that in mind when you read today's numbers.

## Temporal ensembling
Query the policy EVERY step. At time t you have up to H predictions for the current action
(from the chunks made at t, t−1, …, t−H+1). Average them with weights `w_i = exp(−m · i)`,
i = 0 for the OLDEST prediction (ACT's convention: older predictions weigh more → smoother).
Cost: one inference per control step. Benefit: smooth motion, no chunk-boundary jumps.

## Today's build (`w6d1_act.py`)
`ACTPolicy` (encode, decode, forward, act with z=0 or sampled z), `act_loss`, `train_act`,
`TemporalEnsembler`.

Run: `uv run pytest weeks/w6-robot-learning-eval/d1-act`
