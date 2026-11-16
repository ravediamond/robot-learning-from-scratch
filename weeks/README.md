# The weeks, in order

One folder per week. **Open the week's `README.md` and follow it**: it holds the full lesson for the five days,
the order of work, and every command. Nothing outside the week folder is required for weeks 1–9.

## Part 1: the core (9 weeks, 1 hour a day)

| Week | Open this | You end the week able to | Needs |
|---|---|---|---|
| 1 | [`w1-pytorch-basics/README.md`](w1-pytorch-basics/README.md) | write a full training script for a small robot policy from a blank file | Python |
| 2 | [`w2-training-real-data/README.md`](w2-training-real-data/README.md) | load episodic data, train stably, encode images, read your own LeRobot dataset | week 1 |
| 3 | [`w3-transformers-gpt2/README.md`](w3-transformers-gpt2/README.md) | write a transformer and load GPT-2's real weights into it | weeks 1–2 |
| 4 | [`w4-generative-flow/README.md`](w4-generative-flow/README.md) | write a VAE, a diffusion model and a flow-matching policy; evaluate in closed loop | weeks 1–2 |
| 5 | [`w5-vlm-vla/README.md`](w5-vlm-vla/README.md) | build a small VLA that follows an instruction | weeks 3–4 |
| 6 | [`w6-robot-learning-eval/README.md`](w6-robot-learning-eval/README.md) | evaluate policies with intervals and paired tests; ACT; DAgger | weeks 1, 4 |
| 7 | [`w7-jepa-world-models/README.md`](w7-jepa-world-models/README.md) | train a JEPA world model from pixels, probe it, plan with it | weeks 1–2, 6 |
| 8 | [`w8-robotics-register/README.md`](w8-robotics-register/README.md) | do kinematics, camera geometry, control; re-derive your own published numbers | weeks 1, 6 |
| 9 | [`w9-mock-interviews/README.md`](w9-mock-interviews/README.md) | pass timed coding, debugging, research and design rounds | weeks 1–8 |

## Part 2 (see `../ROADMAP.md`)

These weeks have a short week page and one-page day lessons; read the primer named on the week page before day 1.

| Week | Open this | Topic |
|---|---|---|
| 10 | [`w10-reinforcement-learning/README.md`](w10-reinforcement-learning/README.md) | returns, PPO, SAC, offline RL, RECAP-style conditioning |
| 11 | [`w11-rewards-sim-sim2real/README.md`](w11-rewards-sim-sim2real/README.md) | reward models, MuJoCo basics |
| 12 | [`w12-eval-harness/README.md`](w12-eval-harness/README.md) | the evaluation harness behind the flagship comparison |
| 13 | [`w13-robot-data/README.md`](w13-robot-data/README.md) | robot data engineering |
| 14 | [`w14-edge-deployment/README.md`](w14-edge-deployment/README.md) | deployment on the robot's computer |

## Inside a week folder

```
weeks/w1-pytorch-basics/
  README.md                  the lesson for the whole week: read the day's section each day
  d1-tensors-shapes/
    predict.md               1. guess, before reading
    README.md                   one-page summary of the lesson
    predict_run.py           3. check your guesses
    w1d1_tensors.py          4. the exercise: you write the code
    test_w1d1_tensors.py        its tests
    review_w1d1.py           5. code with one planted bug: you fix it
    test_review_w1d1.py         its test
    questions.md             6. answer out loud
  d2-…/  d3-…/  d4-…/  d5-…/
```
Step 2 is reading the day's section of the week `README.md`.
