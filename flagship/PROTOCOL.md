# Protocol v1 (edit, then commit BEFORE the first rollout; every change after that gets a new version)

## Setup
- SO-101 follower, leader arm unplugged during evaluation. Arm calibration file: path + hash in the manifest.
- Cameras: same models, mounts and exposure settings as in the training data (fixed exposure, no auto white balance).
- Lighting: artificial, curtains closed. Table covered with the same mat as in training.
- Compute: Jetson Orin in a fixed power mode (`nvpmodel` mode and `jetson_clocks` recorded), nothing else running.
- Objects: one cube and one bowl from the training set; a spare identical cube, swapped every 50 rollouts (logged).

## Start configurations
- 10 cube positions marked with tape on a 3 × 4 grid inside the reachable area (2 grid points left empty: the
  extreme corners), bowl fixed at one marked position. Positions numbered 0–9, photo of the map committed.
- Arm start pose: the recorded rest pose, verified before each rollout.

## Schedule and blinding
- `make_sheet.py` generates the run sheet: blocks of one start with every policy once, rotating order.
- The operator sees aliases A–D only. `aliases_SEALED.json` stays closed until the analysis.
- Policies are launched by alias from a script; logs and video files are named by trial number, not policy.
- 2 warm-up rollouts per policy per session (start 0), declared here, never analysed.

## One rollout
1. Reset the scene to the start configuration from the sheet; check the arm pose.
2. Start the policy (alias), start the timer.
3. Stop at success, at a safety stop, or at the **timeout of 60 s**.
4. Record immediately (schema below), then reset.

## Outcomes (decided by the operator, video kept)
- **success**: the cube is resting inside the bowl and the gripper has released it, within 60 s.
- **failure**: anything else with the rollout run to the end; exactly one primary `failure_tag`:
  - `perception`: the arm goes to a wrong place (no attempt at the cube's real position)
  - `grasp`: closes on nothing, or the cube slips before lift
  - `transport`: cube dropped between lift and bowl
  - `placement`: cube released outside the bowl, or bounces out
  - `collision`: hits the bowl, the table hard, or knocks the cube away before grasping
  - `timeout`: still trying at 60 s with none of the above
  - `safety_stop`: operator stops the arm for safety
- **aborted**: the rollout could not be run fairly (hardware fault, wrong start, someone bumped the table).
  Re-run at the end of the session with the same start and policy; report the count.
- `stage_reached`: number of stages completed among reach (gripper within 3 cm above the cube), grasp
  (cube between the fingers when lift starts), lift (cube off the table), place (cube in the bowl).
- `interventions`: number of times the operator touched the scene or the arm during the rollout (should be 0).

## Records (`records.csv`, one row per rollout, validated when written)
`trial, alias, start_id, outcome, failure_tag, stage_reached, duration_s, interventions, operator, notes`

## Second rater
A second person labels outcome and failure tag from video for 20% of the rollouts (every fifth trial),
blind to the alias. Report Cohen's κ for the failure tags.

## Latency
Measured on the Orin in the evaluation configuration, per policy: 50 warm-up calls, then 500 timed calls
of the full observation-to-action path (preprocessing + model + postprocessing), GPU synchronised before
each stop. Execution settings (chunk length, actions executed per chunk, sync/async, flow steps) recorded per policy.

## Manifest (one JSON per session)
git SHA of the eval code, checkpoint fingerprints, dataset name + revision, resolved policy configs and
hashes, execution settings, hardware (Orin model, power mode, camera serials), package versions (torch,
CUDA, LeRobot), protocol version, operator.

## Stop rules
- Fixed n: 50 analysed rollouts per policy. No early stopping on results.
- A session stops if the frozen reference (ACT checkpoint, alias rotated) fails 3 consecutive rollouts on
  starts it usually succeeds on: check the hardware, log it, resume.
