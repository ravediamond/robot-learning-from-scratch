# 12. Simulation and sim-to-real

Week 11 (D2; D3–D5 on the roadmap). What a robot research engineer must be able to say about simulators
even when the job is real-robot work.

## 1. The objects
MuJoCo: `MjModel` (constants), `MjData` (state), `mj_step` (advance), `mj_forward` (compute positions,
Jacobians, forces without advancing). Physics at 500–1000 Hz; policies at 10–50 Hz: one control step =
`round(control_dt / timestep)` physics steps; choose a timestep that divides the control period.

## 2. Control in sim
Torque motors + your PD, or position actuators that model servos (kp, kv, armature, damping, friction,
force limits). Gravity compensation adds `qfrc_bias`. IK by damped least squares on `mj_jacSite`.
The SO-101's servos have internal PID loops, delays and deadbands that a naive model does not have.

## 3. The reality gap
Dynamics (friction, backlash, servo response, payload), latency (camera, inference, bus), sensing
(noise, exposure, calibration), and contact. Tools: **system identification** (fit simulator parameters
to recorded real trajectories: CEM or gradient-based, on chirp or step inputs), **domain randomisation**
(train or tune over a distribution of parameters wide enough to contain reality), **latency modelling**,
and **real-to-sim** (fit the scene from the real setup).

## 4. Proving sim means something
Sim-to-sim first (train in one simulator, test in another), then a sim–real correlation study: rank
policies in sim and on the robot with the same protocol; report rank correlation and MMRV (mean
maximum rank violation). A simulator is a valid evaluator only for the policies and tasks where that
correlation was shown.

## 5. Choosing a simulator
MuJoCo (accurate rigid-body contact, CPU, great for one robot and system ID), MuJoCo MJX / Warp (same
models on GPU), Isaac Lab (GPU-parallel RL, rendering, large ecosystem), Genesis (fast, younger).
Use the official SO-101 MJCF from `TheRobotStudio/SO-ARM100` for real work.

## Questions
1. `int((1/30) / 0.002)`: why is it a bug in a simulator loop?
<details><summary>Answer</summary>16.67 truncates to 16: each control step simulates 32 ms instead of 33.3 ms, so simulated time runs 4% fast against the robot.</details>

2. Steady-state error of a PD joint under gravity, and two fixes?
<details><summary>Answer</summary>e = g(q)/kp. Gravity compensation (needs a mass model) or an integral term (lag, wind-up).</details>

3. Domain randomisation vs system identification?
<details><summary>Answer</summary>SysID narrows the distribution to the real robot (accurate when the model class is right); randomisation widens it so that reality is inside (robust, can over-regularise). Use both: identify what you can, randomise what you cannot.</details>

4. How do you know a simulator ranks policies like the real robot?
<details><summary>Answer</summary>Evaluate several policies in both with the same protocol, report Spearman/Kendall and MMRV with uncertainty; trust it only within that set of tasks and policies.</details>
