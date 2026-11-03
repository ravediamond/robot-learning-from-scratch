# 09. LeRobot reading map (v0.6.2)

Mapped from the source at LeRobot `0.6.2`, commit `06f78b6` (2026-11-03). Paths are under
`src/lerobot/`. LeRobot moves fast: before trusting a line here, check the version your code pins
(`pip show lerobot`, or the Docker tag on the Orin). Older versions differ in two places that matter
for reading: normalisation lived INSIDE the policy, and logging used wandb (see the end).

Not first principles: plumbing. Read it once with this map open, then read real runs with it.

## 1. Where a run starts
| Command | File | Entry |
|---|---|---|
| `lerobot-train` | `scripts/lerobot_train.py` (1,076 lines) | `train(cfg)` at line 529; one step = `update_policy` (line 181) |
| `lerobot-record`, `-replay`, `-teleoperate` | `scripts/lerobot_record.py` … | data collection |
| `lerobot-rollout` | `scripts/lerobot_rollout.py` + `rollout/` | runs a policy on the robot; `--inference.type=sync\|rtc\|remote` |
| `lerobot-policy-server` | `scripts/lerobot_policy_server.py` + `remote_inference/` | serves one checkpoint to one robot client |
| `lerobot-eval` | `scripts/lerobot_eval.py` | sim evaluation |

Reading order for `train()`: validate cfg → accelerator → `set_seed(cfg.seed)` (line ~574) → datasets →
policy (`make_policy`) → processors (`make_pre_post_processors`) → resume phase 1 (step + RNG) →
dataloaders → `accelerator.prepare` → resume phase 2 (weights, optimizer, scheduler, EMA) → loop
(`update_policy`, log every `log_freq`, `save_checkpoint` every `save_freq` and at the end).

## 2. Configs (draccus dataclasses, CLI = dotted fields)
- `TrainPipelineConfig` (`configs/train.py`): `output_dir`, `resume=False`, `seed=1000`, `batch_size=8`,
  `steps=100_000`, `save_freq=20_000`, `log_freq`, `use_policy_training_preset=True`.
- With the preset on, the optimizer and scheduler come from the POLICY config
  (`get_optimizer_preset()` / `get_scheduler_preset()`, e.g. SmolVLA: AdamW + its own lr, betas, grad clip).
  Reading trap: the lr you see on the command line may not be the one used. Check `train_config.json`.
- Policy configs: `policies/<name>/configuration_<name>.py`, subclasses of `PreTrainedConfig`
  (`configs/policies.py`). Fields to find first: `chunk_size`, `n_action_steps`, `n_obs_steps`,
  `normalization_mapping`, the optimizer fields.
- `--policy.type=smolvla` builds a fresh config; `--policy.path=<hub id or dir>` loads a pretrained one
  (its `config.json`), then CLI overrides apply.

## 3. Data: how a sample is built
- `datasets/lerobot_dataset.py` + `datasets/dataset_reader.py`. Parquet + mp4 + `meta/` (you did this by
  hand in W2 D4).
- Windows come from **`delta_timestamps`**, resolved from the policy config's `*_delta_indices`
  (`datasets/factory.py: resolve_delta_timestamps`). For SmolVLA, `action_delta_indices = range(chunk_size)`:
  each sample carries `action` of shape (chunk_size, D).
- Past the episode end the window is padded and flagged with **`action_is_pad`** (`dataset_reader.py`,
  `delta_window`). The policy's loss must mask it: check that it does (W1 D2 review bug).
- Frames are decoded from video on the fly: the usual bottleneck (W2 D5).

## 4. Normalisation lives in PROCESSORS, not in the model
- Pipelines: `processor/pipeline.py` (`PolicyProcessorPipeline`), built by
  `policies/factory.py: make_pre_post_processors` from `policies/<name>/processor_<name>.py`.
- SmolVLA preprocessor: rename → add batch dim → newline task → tokenizer → to device → **normalize**.
  Postprocessor: **unnormalize** → to CPU.
- `processor/normalize_processor.py`: `NormalizerProcessorStep` / `UnnormalizerProcessorStep`. Modes
  (`configs/types.py: NormalizationMode`): `MEAN_STD`; `MIN_MAX` (to [-1, 1] from min/max); `QUANTILES`
  (q01/q99 to [-1, 1]); `QUANTILE10` (q10/q90); `IDENTITY`.
- Defaults: SmolVLA VISUAL=IDENTITY, STATE=MEAN_STD, ACTION=MEAN_STD. ACT: all MEAN_STD.
- Stats: from `dataset.meta.stats` at the start of training; saved WITH the checkpoint as
  `policy_preprocessor_step_*.safetensors`; on resume the checkpoint's stats win.
- Reading consequence: inference must load the processors from the SAME checkpoint. A policy loaded without
  its postprocessor emits actions in normalised units (W8 D5, W9 D3 bug 6).
- Old checkpoints (normalisation inside the model as buffers) are converted by
  `processor/migrate_policy_normalization.py`. If code you read calls `policy.normalize_inputs`, it is pre-processor
  LeRobot.

## 5. The policy contract (`policies/pretrained.py`, class `PreTrainedPolicy`)
| Method | Used by | Returns |
|---|---|---|
| `forward(batch)` | training | `(loss, info_dict)` |
| `predict_action_chunk(batch)` | async / RTC / server | `(B, chunk_size, D)` normalised actions |
| `select_action(batch)` | sync control loop | ONE action `(B, D)` |
| `reset()` | start of every episode | clears the action queue |
| `from_pretrained` / `_save_pretrained` | load / save | `config.json` + `model.safetensors` |

**The queue** (SmolVLA `modeling_smolvla.py` line 256; ACT `modeling_act.py`): `select_action` calls the
model only when the queue is empty, enqueues the first `n_action_steps` of the chunk, then pops one per
call. So `n_action_steps` = how many actions run open-loop between inferences. Forgetting `reset()` between
episodes executes the previous episode's leftover actions. ACT with `temporal_ensemble_coeff` set runs
temporal ensembling instead (inference every step, W6 D1). SmolVLA's flow-matching path:
`embed_prefix` (line 604) / `embed_suffix` (698) / `sample_actions` (776) / `denoise_step` (827), with
`make_att_2d_masks` building the block masks you wrote in W5 D2.

## 6. One training step (`update_policy`)
`accelerator.accumulate` → `accelerator.autocast` → `policy.forward` → backward → clip to the optimizer's
`grad_clip_norm` → `optimizer.step` → `scheduler.step` (per micro-batch) → EMA (if any). Under fp16 the
GradScaler can skip steps on overflow.

## 7. Checkpoints and resume (`common/train_utils.py: save_checkpoint` line 217)
```
<output_dir>/checkpoints/<step>/pretrained_model/  config.json, model.safetensors, train_config.json,
                                                   policy_{pre,post}processor.json + *_step_*.safetensors
<output_dir>/checkpoints/<step>/training_state/    optimizer_state.safetensors, optimizer_param_groups.json,
                                                   rng_state.safetensors, scheduler_state.json, training_step.json
<output_dir>/checkpoints/last -> <step>
```
Resume: `--resume=true --config_path=<ckpt>/pretrained_model/train_config.json`. Two phases (step + RNG
before `prepare`, sharded state after). The data order is a function of (seed, epoch) and the sampler
offset is derived from the step: sample-exact only with the same world size and batch size (W9 D1).

## 8. Logging: Trackio, not wandb (in 0.6.2)
`configs/default.py: TrackioConfig`; local SQLite under `~/.cache/huggingface/trackio`, view with
`trackio show --project <p>`. Old configs' `wandb` block is migrated (`configs/train.py`,
`_migrate_legacy_wandb_section`). Code or docs that talk about wandb are older than this version.

## 9. Inference on the robot
- `inference/`: backends `sync` (the policy's own `select_action`), `rtc` (local async worker with RTC),
  `remote` (policy server over Zenoh). Config: `--inference.type=…` on `lerobot-rollout`.
- `policies/rtc/`: RTC guidance, action queue, latency tracker. `inference/rtc.py`: the engine side.
- `remote_inference/README.md`: the design reference (ownership, chunk contract, failure handling). Read §1–2.

## 10. Checklist when reading a LeRobot run someone (or Claude Code) did
1. Version: which LeRobot, which commit, which Docker image?
2. `train_config.json`: policy type/path, `chunk_size`, `n_action_steps`, normalisation mapping, lr and
   scheduler actually used (preset or not), steps, batch, seed.
3. Dataset: repo id + revision; `delta_timestamps`; is `action_is_pad` masked in the loss?
4. Processors: saved next to the weights? Loaded at inference from the same directory?
5. Load: strict? Count of tensors? (W8 D5, your W1 audit.)
6. Inference: which backend, which `n_action_steps`, `reset()` per episode, measured latency vs control rate.
7. Results: from which checkpoint step, on which episodes, with which seed: re-derivable (W8 D4)?

## Questions
1. Where exactly is the action un-normalised on the way from the model to the motors, in 0.6.2?
2. `n_action_steps = 50`, `chunk_size = 50`, 30 Hz, 240 ms inference, sync backend: draw the timeline. What does the robot do during inference?
3. You resume a run on 2 GPUs that was started on 1. What does LeRobot warn about, and what is not exact any more?
4. Your harness pins an older LeRobot. Name two things in this map that would read differently there.
