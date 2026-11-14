# W11 D2: MuJoCo basics on an SO-101-like arm

Primer: `knowledge/12-simulation-sim2real.md` §1–2. Docs: mujoco.readthedocs.io (Programming → Simulation).

## The objects
- `MjModel`: constants (masses, joint axes, timestep, actuator limits). `MjData`: state (`qpos`, `qvel`,
  `ctrl`, `time`) and everything computed from it.
- `mj_step(model, data)`: advance by `model.opt.timestep` (here 2 ms). `mj_forward`: compute positions,
  Jacobians and forces for the current state WITHOUT advancing time.

## Two rates
Physics runs at 500 Hz; your policy at 30 Hz (the SO-101 camera rate). One control step = hold the target for
`round(control_dt / timestep)` physics steps. `(1/30) / 0.002 = 16.67`: `int()` truncates to 16 and your
simulated clock runs 4% fast against the real robot. Better still, pick a timestep that divides the
control period, and assert it.

## PD control and gravity
The arm has torque motors. A position servo is `τ = kp (q_des − q) − kd q̇`. Under gravity a pure PD
controller has a steady-state error `≈ τ_gravity / kp`. Adding `data.qfrc_bias` (gravity + Coriolis) to the
torque removes it: **gravity compensation**.

## Jacobian and IK
`J = ∂(end-effector position)/∂q`, shape (3, nv), from `mj_jacSite`. Damped least squares IK:
```
dq = Jᵀ (J Jᵀ + λ² I)⁻¹ (target − ee)
```
λ keeps the step finite near singularities (arm stretched out).

## Today's build (`w11d2_mujoco.py`)
`n_substeps`, `pd_torque`, `control_step`, `ee_pos`, `ee_jacobian`, `ik_step`, `solve_ik`.

Run: `uv run pytest weeks/w11-rewards-sim-sim2real/d2-mujoco-basics`
See it move (optional, needs a display): `uv run python -m mujoco.viewer --mjcf=weeks/w11-rewards-sim-sim2real/d2-mujoco-basics/so101_like.xml`
