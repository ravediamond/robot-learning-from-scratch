# W10 D1: Returns, advantages, and the policy gradient

Primer: `knowledge/11-reinforcement-learning.md` §1–2.

## The three objects
**Discounted return** from step t, cut at episode ends (`done_t` = the episode ended AT step t):
```
G_t = r_t + γ (1 − done_t) G_{t+1}
```
**Policy gradient (REINFORCE)**: raise the log-probability of actions in proportion to what followed.
```
∇J = E[ ∇ log π(a_t | s_t) · (G_t − b(s_t)) ]        loss = −mean( log π(a_t|s_t) · (G_t − b_t).detach() )
```
Subtracting a baseline `b` that does not depend on the action leaves the gradient unbiased and cuts its
variance. With `b = V(s)` the weight is the **advantage** A = G − V: "better or worse than usual here".

**GAE(γ, λ)**: a dial between low variance (λ=0: one-step TD error) and low bias (λ=1: full return − V).
```
δ_t = r_t + γ (1 − done_t) V_{t+1} − V_t          A_t = δ_t + γ λ (1 − done_t) A_{t+1}
```
`V_{T}` after the last stored step is `last_value` (the bootstrap).

## Today's build (`w10d1_pg.py`)
`discounted_returns`, `gae`, `GaussianPolicy.log_prob`, `reinforce_loss`, `train_reinforce` on `Reach2D`.
Tensors are time-major: `(T, N)` = T steps of N parallel envs.

Run: `uv run pytest weeks/w10-reinforcement-learning/d1-returns-policy-gradient`
