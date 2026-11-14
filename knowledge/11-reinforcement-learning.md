# 11. Reinforcement learning for robot policies

Week 10 and W11 D1. Robot research-engineer loops screen RL next to imitation learning: the frontier
recipe is "imitate first, then improve from the robot's own experience". Know the mechanisms, the
failure modes, and why most real-robot RL is off-policy, offline, or advantage-conditioned.

## 1. Returns and the policy gradient
Return `G_t = r_t + γ (1 − done_t) G_{t+1}`. Policy gradient: `∇J = E[∇ log π(a|s) (G − b(s))]`.
A baseline that does not depend on the action keeps it unbiased and cuts variance; with `b = V(s)` the
weight is the advantage. Time limits are not terminal states: bootstrap through them.

## 2. Advantages: GAE
`δ_t = r_t + γ(1 − d_t)V_{t+1} − V_t`, `A_t = δ_t + γλ(1 − d_t)A_{t+1}`. λ = 0: TD error (low variance,
biased by V). λ = 1: Monte-Carlo return minus V (unbiased, noisy). λ ≈ 0.95 in practice.

## 3. PPO
`L = −E[min(ρA, clip(ρ, 1 ± ε)A)]`, ρ = π_new/π_old. Pessimistic: no gain from moving ρ beyond the clip in
the helpful direction. Store log π_old at collection; normalise advantages per minibatch; several epochs.
The default for GPU-parallel simulation (locomotion, Isaac Lab), wasteful on real hardware.

## 4. SAC (off-policy)
Replay buffer, twin Q critics with target networks, tanh-squashed Gaussian actor with the log-det
correction `−Σ log(1 − tanh²u)`, learned temperature towards a target entropy. Target:
`y = r + γ(1 − d)(min Q'(s', a') − α log π(a'|s'))`. Real-robot variants (HIL-SERL) add demos in the
buffer, human interventions, a learned success reward, high update-to-data ratios with regularised critics.

## 5. Offline RL
Fixed data, no exploration. Naive Q-learning queries Q at unseen actions and exploits extrapolation
errors. IQL never does: V by **expectile** regression on dataset actions (τ > 0.5 leans to the best),
Q from r + γV(s'), policy by **advantage-weighted regression** on dataset actions. Baseline to beat:
filtered BC.

## 6. RL on top of imitation for VLAs (RECAP, π*0.6)
Train a value function on demos + rollouts + interventions; per-step advantage → binary "better than
usual" indicator with a **per-task** threshold; train the VLA on ALL data conditioned on the indicator
(with dropout); ask for "good" at test time. Supervised loss, no policy gradient through a 3B model.

## 7. Rewards
Success classifiers (sparse, cheap, threshold for precision: RL exploits false positives), progress
models from ordering inside episodes (Bradley–Terry pairs), trajectory-comparison reward models
(Robometer), value functions, VLM-based progress estimates. Validate any reward before spending robot
hours: precision on hard negatives, rank correlation with human progress, and a short closed-loop pilot.

## Questions
1. Why does subtracting a baseline not bias the policy gradient?
<details><summary>Answer</summary>E_a[∇log π(a|s) b(s)] = b(s) ∇Σ_a π(a|s) = b(s)∇1 = 0.</details>

2. Why the `min` in PPO?
<details><summary>Answer</summary>It takes the pessimistic of the clipped and unclipped objectives, so the policy gets no reward for moving the ratio past 1 ± ε in the helpful direction, while harmful moves are always penalised.</details>

3. What breaks if you forget the tanh correction in SAC?
<details><summary>Answer</summary>log π is the density of the pre-squash sample, not of the action; entropy near the bounds is overestimated, the temperature and the actor optimise the wrong quantity, and actions saturate.</details>

4. Why is naive off-policy Q-learning unstable on a fixed dataset?
<details><summary>Answer</summary>The bootstrap uses Q at actions absent from the data; the max picks their over-estimates and nothing corrects them without new data.</details>

5. 10,000 h teleop + 100 h autonomous rollouts: one-paragraph plan.
<details><summary>Answer</summary>Imitate on curated teleop; label rollouts (success, progress) and train a value function; use interventions as corrections, advantage-condition the policy on rollouts (RECAP-style), train failure detectors; explicit mixture weights so the 100 h count; paired real-robot evaluation.</details>
