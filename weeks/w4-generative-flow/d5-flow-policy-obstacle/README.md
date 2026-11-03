# W4 D5: A flow-matching action head vs an MSE head, closed loop (capstone, part 2)

Book: 11.1–11.3. World: `sim2d.py` (read it: 60 lines). Demos go around the obstacle, half above,
half below. Observation = position (2). Action chunk = next H = 8 velocities.

## The experiment
1. Two policies, same data, same chunk size, same size of MLP:
   - **MSE head**: obs → chunk, trained with masked MSE.
   - **Flow head**: obs + noisy chunk + t → velocity, trained with flow matching; sample with 10 Euler steps.
2. Scale actions to units of `MAX_STEP` for training (values ~[−1, 1]); multiply back to act.
3. **You write the rollout**: replan every `H_exec = 4` steps from the current position, clip every
   action, stop an episode at collision or success, `T_MAX` steps at most. Batched over episodes.
4. Report collision and success rates with Wilson intervals (`sim2d.wilson`), 200 episodes each.

Prediction to write down first: which head collides more, and by how much?

## Today's build (`w4d5_flow_policy.py`)
`MSEChunkPolicy`, `FlowChunkPolicy` (both with `.act(obs, generator) -> raw actions (N, H, 2)`),
`masked_mse`, `train_mse_policy`, `train_flow_policy`, `rollout`.
Test bars: MSE collision rate > 30%; flow collision < 15% and success > 85%. (~1 min)

Run: `uv run pytest weeks/w4-generative-flow/d5-flow-policy-obstacle -s`
