# Runbook: from harness to 200 real rollouts

Written against LeRobot 0.6.x. Command names and flags change between versions: check each one with
`--help` on your install before a session. Nothing here was executed on hardware by the author of this
file; the analysis scripts were tested on the synthetic data of `demo_records.py`.

## Before the first session
1. Labs green: `uv run pytest weeks/w12-eval-harness`. Dry run of the report on synthetic data (README step 3).
2. Train or fetch the four checkpoints on the SAME dataset revision. For each, write down: checkpoint path,
   fingerprint (`tensor_fingerprint` on the loaded state dict, W12 D5), execution settings.
3. Fill and commit `PROTOCOL.md` and `PREREG.md`. Tape the start map, photograph it, commit the photo.
4. Calibrate the arm, record the calibration file hash.

## Latency probe on the Orin (per policy, once per setting)
- Fix the power mode (`sudo nvpmodel -m <mode>`, `sudo jetson_clocks`), note them.
- Load the policy exactly as the evaluation will, build one real observation (camera frames + state).
- Time the full observation-to-action call with your W12 D4 `measure` (`clock=time.perf_counter`), 50 warm-up,
  500 timed; inside the timed function call `torch.cuda.synchronize()` before returning.
  For chunked policies time the call that produces a NEW chunk (in LeRobot, `predict_action_chunk`;
  `select_action` returns queued actions without inference most of the time and would hide the cost).
- Append `policy,ms` rows to `latency.csv`; write `settings.json` with chunk, exec_steps, hz.

## A session
1. `python flagship/make_sheet.py --policies act smolvla vla_jepa mine --starts 10 --repeats 5 --seed <n> --out flagship/sessionN`
   (one sheet per session if you split the 200 rollouts; use a different seed each time and log it).
2. Write the session manifest (W12 D5 `run_manifest`) before the first rollout.
3. Warm-up rollouts as declared. Then follow the sheet in order. Launch by alias from a script that maps the
   alias to the checkpoint without printing the policy name.
4. Running a policy on the arm: LeRobot records evaluation episodes with `lerobot-record` and a
   `--policy.path` pointing at the checkpoint (episodes go to an `eval_` dataset, which also keeps the video).
   One episode per sheet row; set the episode time to the protocol timeout and the reset time long enough
   for your reset.
5. After each rollout, fill the row of `records.csv` (outcome, tag, stage, duration, interventions).
6. Every 50 rollouts: battery/servo temperature check, cube swap, short break. Log both.

## After the session
1. Second rater labels every fifth trial from video; compute κ with W12 D3 `cohen_kappa`.
2. `python flagship/analyze.py --records ... --aliases flagship/sessionN/aliases_SEALED.json --latency ... --settings ... --out report.md`
3. Write deviations in `PREREG.md` (append only). Only then look at which alias was which policy in the report.

## The page
- The generated report (tables) + 6 short videos (one success and one typical failure for three policies).
- The protocol, the pre-registration with its commit hash, the manifests, the code link.
- One paragraph: what the experiment can and cannot conclude (the minimum detectable difference).
