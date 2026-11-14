# Week 14: deploying policies on the robot's computer

Primer: `knowledge/15-edge-deployment.md`. Builds on W8 D3 (latency, chunk switching) and W12 D4 (latency on target).

A policy that wins in a notebook still has to run at 30 Hz on a Jetson next to a camera, a servo bus and
a human. This week covers the path from checkpoint to arm: export with numerical parity, quantisation
you can defend, real-time chunking so the arm does not jerk between chunks, a safety layer between the
policy and the motors, and the rollout of a new model to a fleet without betting the fleet on it.

| Day | Lab | Review bug |
|---|---|---|
| d1 | export: a deployable wrapper with normalisation inside, ONNX, parity, dynamic batch | batch axis frozen at 1 |
| d2 | quantisation: per-tensor vs per-channel int8, calibration, fake-quant with a straight-through estimator | no straight-through gradient |
| d3 | real-time chunking: shifting the previous chunk, soft masks, guided flow sampling | previous chunk not shifted |
| d4 | the safety layer: limits, rate limits, stale-observation watchdog, NaN guard, e-stop latch | staleness from receipt time |
| d5 | fleet rollout: artifact checks, shadow mode, canary with a rollback test + **design** | counts compared instead of rates |

ONNX labs need `onnx` and `onnxruntime` (in `pyproject.toml`). TensorRT on the Orin is in `d1/tensorrt_on_orin.md`
(not runnable on a Mac).
