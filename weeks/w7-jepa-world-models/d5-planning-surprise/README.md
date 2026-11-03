# W7 D5: Planning with your world model, and surprise (capstone, part 2)

Book: 12b (JEPA planning and grounding), 12.4 (three roles), 13.2. Primer 06. Coding list item 23.
Uses YOUR `../d4-mini-lewm/w7d4_lewm.py` (imported automatically) and `probes_ref.py`.

## Role 1, planner: CEM in latent space (MPC)
```
z0 = enc(image now);  cost(z) = how far the predicted latent is from the goal
mu = 0, std = AMAX                        per episode, (H, 2)
repeat iters: sample N sequences ~ N(mu, std), clamp to [-AMAX, AMAX]
              roll out the predictor H steps; cost = SUM over the H predicted latents
              keep the K lowest-cost sequences (elites); mu, std = their mean, std (+1e-3)
execute mu[0], observe, replan                                   (receding horizon)
```
Two costs, same predictor:
- **latent L2**: `||z_h − enc(goal image)||²`
- **probe-decoded**: `||probe(z_h) − goal position||²` (an MLP probe trained on train episodes)
If the second works much better, the dynamics are fine and the problem is the latent METRIC.

## Role 2, evaluator: surprise
`s_t = ||pred(z_t, a_t) − z_{t+1}||²`. Unexpected events (a teleport: someone moved the object) should
spike. Measure AUROC (teleport transitions vs normal ones).

## Today's build (`w7d5_plan.py`)
`cem_plan`, `latent_cost`, `probe_cost`, `run_planner`, `surprise_scores`.
Tests (~1.5 min): latent-L2 planner beats random by 20 points; probe-cost planner > 70%; surprise AUROC > 0.95.

Run: `uv run pytest weeks/w7-jepa-world-models/d5-planning-surprise -s`
