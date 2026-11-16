# W11 D1: Reward models

> **Full lesson for this day:** [`../README.md`](../README.md#day-1-reward-models) (the week page: concepts from zero, examples, traps).
> This file is the one-page summary. Order of work: `predict.md` → the lesson → `predict_run.py` → the exercise → the review file → `questions.md`.

Primer: `knowledge/11-reinforcement-learning.md` §7. Papers: HIL-SERL (Luo et al., 2024, success classifiers);
Robometer (in your project files, trajectory comparisons); GVL / VLM progress estimators.

## Why
On a real robot nobody hands you `reward = -distance`. Three practical sources:
1. **Success classifier**: a binary model on the final image/state. Sparse, cheap to label.
2. **Progress model**: a scalar that rises along good trajectories. Learned from ORDER only: inside one
   successful episode, a later frame should score higher than an earlier one. No reward labels needed.
3. **Value function** (W10 D5): steps-to-success, trained on successes and failures.

## Two rules that decide whether it works
- **RL exploits false positives.** A classifier that says "success" in a wrong state gives the agent a
  place to collect reward forever. Choose the threshold for **precision**, then accept the recall you get.
- **Compare only what is comparable.** Frame 10 of episode A and frame 40 of episode B are not ordered:
  B may have started closer. Build pairs inside one episode (or between trajectories of the same task
  with known quality, as Robometer does).

## Pairwise (Bradley–Terry) loss
For a pair (a, b) with label y = 1 if b is later: `loss = BCE_with_logits(score(b) − score(a), y)`.

## Today's build (`w11d1_rewards.py`)
`train_success_classifier`, `threshold_for_precision`, `progress_pairs`, `pairwise_loss`,
`train_progress_model`, `spearman`. Data comes from `reward_data.py` (provided).

Run: `uv run pytest weeks/w11-rewards-sim-sim2real/d1-reward-models`
