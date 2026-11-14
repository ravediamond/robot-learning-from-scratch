# 15. Deploying policies on the robot's computer

Week 14. From checkpoint to motors, and from one robot to a fleet.

## 1. Export
One module = normalise → model → unnormalise, stats as buffers. ONNX (legacy exporter `dynamo=False`
for simple graphs), dynamic batch axes, parity on real recorded observations (fp32 ~1e-6), then
TensorRT engines built on the device (fp16; min/opt/max shapes for dynamic axes). Large VLAs: split
backbone (once per observation) and action head (per flow step), cache the prefix.

## 2. Quantisation
Symmetric int8, per-channel for weights, percentile calibration for activations. PTQ first; QAT with
fake quantisation and the straight-through estimator when PTQ hurts. Keep sensitive layers in fp16.
Edge decoding is often bandwidth-bound: weight-only low-bit quantisation helps most there. Judge by
action error in joint units and closed-loop success.

## 3. Real-time chunking
Async inference plus the stale prefix and mode switching at seams. RTC: generate the next chunk as
inpainting conditioned on the previous chunk shifted by the executed steps, with a soft mask over the
overlap; works on any flow/diffusion policy without retraining. Alternatives: temporal ensembling
(regression heads), training with the prefix as input.

## 4. Safety layer
Firmware limits → deterministic filter (NaN guard, joint limits with margins, rate limits, watchdog on
CAPTURE time, latched e-stop) → monitors → operator and procedures. Exhaustively tested, small, logged.

## 5. Fleet rollout
Fingerprint check at load, shadow mode, canary with a minimum exposure and a rate test on
interventions (conditional binomial on Poisson counts), staged rollout, rollback at task boundaries,
desired-state configuration per robot with drift alerts.

## Questions
1. What must be inside an exported policy graph?
<details><summary>Answer</summary>Everything between raw observation and command that training applied: preprocessing, normalisation, model, unnormalisation.</details>

2. Why does QAT need the straight-through estimator?
<details><summary>Answer</summary>Rounding has zero gradient almost everywhere; the STE passes the gradient through as identity so weights can still learn.</details>

3. Two problems RTC solves?
<details><summary>Answer</summary>The new chunk's first actions are stale (in the past) and an independent sample can switch modes: both cause jumps at the seam.</details>

4. Watchdog on receive time: what goes wrong?
<details><summary>Answer</summary>Buffered frames arrive late but look fresh; the policy acts on the past. Use capture timestamps.</details>

5. Canary 3 interventions in 40 h vs fleet 30 in 400 h?
<details><summary>Answer</summary>Same rate (0.075/h); one-sided p ≈ 0.59: continue.</details>
