# W10 D3: SAC, the off-policy workhorse

Primer: `knowledge/11-reinforcement-learning.md` §4. Papers: Haarnoja et al., 2018 (SAC); Luo et al., 2024 (HIL-SERL, SAC on real robots).

## Why off-policy
PPO throws each batch away. On a real arm every transition is expensive, so we keep them in a **replay
buffer** and learn a **Q-function** from all of them. SAC is the standard recipe for continuous actions.

## The five pieces
1. **Tanh-squashed Gaussian policy.** Sample `u ~ N(μ, σ)`, act `a = tanh(u)` (bounded actions).
   The density changes with the squash: `log π(a) = log N(u; μ, σ) − Σ log(1 − tanh²(u))`.
2. **Twin Q target** (take the minimum of two critics to fight over-estimation), with the entropy term:
   `y = r + γ (1 − done) · ( min(Q1', Q2')(s', a') − α log π(a'|s') )`, with `a' ~ π(·|s')`.
3. **Actor loss**: `mean( α log π(a|s) − min(Q1, Q2)(s, a) )`, with `a` re-sampled (reparameterised).
4. **Temperature** α learned so the entropy stays near a target (usually `−act_dim`):
   `loss_α = −mean( log_α · (log π + target_entropy).detach() )`.
5. **Target critics** follow the critics slowly: `θ' ← (1 − τ) θ' + τ θ`.

## Today's build (`w10d3_sac.py`)
`squashed_sample`, `soft_update`, `q_target`, `actor_loss`, `alpha_loss`, `ReplayBuffer`, `train_sac`.

Run: `uv run pytest weeks/w10-reinforcement-learning/d3-sac`
