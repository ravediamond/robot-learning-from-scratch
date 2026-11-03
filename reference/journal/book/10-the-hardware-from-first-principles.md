# Chapter 10. The hardware, from first principles

A research engineer who cannot say what happens between a policy's output tensor and a servo's shaft will be found out in the first ten minutes. (The general concepts — kinematics, control loops, camera geometry, action spaces — are explained from zero in chapter 10a; the data format and pipeline in chapter 10b.) This chapter is the physical stack of your setup, bottom up, with the numbers, so that every later chapter's claims ("position-controlled", "238 ms per chunk", "the gripper reads 1.1° on a rim") have a mechanism behind them.

## 10.1 The arm: SO-101

Six degrees of freedom: shoulder pan, shoulder lift, elbow flex, wrist flex, wrist roll, gripper. Each joint is one **Feetech STS3215** serial servo: a DC motor, a reduction gearbox, a magnetic position encoder (12-bit, 4 096 counts per turn, ~0.088°), and a microcontroller running a position loop. Two arms: the **follower** (the robot) and the **leader** (the teleop handle, same servos, read-only).

**Position control.** You send each servo a goal position (and optionally speed and acceleration limits); the servo's own PID loop drives the motor to that position and holds it. You do not command torque or velocity directly. Consequences that run through the whole repository:

- The commanded joints *are* the near-future arm state, up to the servo's tracking lag and saturation. That is why the commanded-joint baseline wins (chapter 3).
- There is no force or torque sensing. The only contact signal is indirect: the position error between goal and present (the servo "pushing" against something), the motor current if the bus exposes it at a usable rate (the CoPRE paper's route), and the gripper aperture.
- Compliance is set by the servo's P gain and torque limit, not by the policy. A stiff servo meeting a rigid obstacle strips gears; the ASPIRE-derived rule "block the arm with a soft hand or foam, never a rigid object" exists for this.
- `STS3215SafetyGuard` in the harness clamps goal positions to the calibrated range and floors the gripper command (at 12 in its units), which is why policies that close "to ~0.1" never fully close through the guard, and why the gripper reads ~1.1° on a bowl rim while the guard's floor is far above that (G15-v3 finding).

**The bus.** Servos are daisy-chained on a half-duplex TTL serial bus at 1 Mbaud. One USB-serial adapter per arm. A read of all six present positions and a write of six goals is one bus transaction each; the LeRobot driver does this at the control frequency. Practical ceiling around 30–60 Hz per arm for read+write; your control loop runs at 30 Hz (33.3 ms tick). Reading extra registers (current, temperature, load) costs bus time; that is the constraint on current-based contact detection.

**Calibration.** Each servo's encoder zero is arbitrary; calibration maps raw counts to joint angles and sets the safe range. The calibration files in `shared/gate/configs/calibration/` are per-arm and must match the arm; a mismatch is a silent wrong-pose bug (the kinematics-observation-order bug you filed upstream was of this family).

**Kinematics.** The URDF in `shared/robot/so101_new_calib.urdf` gives link lengths and joint axes; forward kinematics maps six angles to a gripper pose; inverse kinematics the reverse (used in G16-lite's repair: hover above the blue centroid → IK → goal joints). The SO-101's reach is ~35 cm; the LEGO cell's layout is constrained by it.

**Teleop.** The leader arm is read at 30 Hz; its joint angles are written to the follower as goals, with the follower's present positions and the cameras recorded. A teleop episode is therefore (leader angles as `action`, follower angles as `observation.state`, two images) per frame. The 1.6 s idle at the start of early recordings (operator settling) is what taught TurboVLA to freeze.

## 10.2 Cameras

Two USB cameras: a Logitech C920 above the table (oblique, not zenithal, because the mount could not go higher) and a small wrist camera on the gripper. Each is read by OpenCV at 30 fps; frames are JPEG/H.264-encoded into the dataset.

What the geometry implies:

- The top view sees nesting and relative position (enough for a two-object relation judge); it does not see occluded grasps well. The wrist view sees the grasp and little else.
- USB webcams have independent, jittering frame timestamps. Two-camera world models (W1) see a few ms of misalignment between views per frame; the W1-A write-up names "USB webcam sync" as a scale-shift risk on surprise thresholds. Hardware-synchronised cameras would remove it; you do not have them.
- Lighting: a window beside the table; curtains closed during sessions; the CF-JEPA angle exists because demos will not control lighting.
- Resolution: policies consume 224×224 or 256×256 crops; the judge works on the full top frame. The LEGO perception overlay (SEVO/ARRO) runs at the full frame once per unit action.

## 10.3 The robot computer: Jetson Orin NX 16 GB

An ARM CPU (8 cores) and an Ampere GPU (1 024 CUDA cores, 32 tensor cores) sharing 16 GB of LPDDR5 memory (unified: the GPU has no separate VRAM, and `getrusage` RSS does not count CUDA buffers, which is why the vla.cpp benchmark's memory column was unreliable). JetPack 6.2, CUDA 12.6, L4T R36.4.3. Power mode MAXN_SUPER. Headless; Docker images for everything (the public `lerobot-jetson` image is the one you built and merged upstream).

The numbers to know, all measured here:

| Thing | On the Orin | Note |
|---|---|---|
| SmolVLA 450M, PyTorch | 974 ms per 50-action chunk | too slow: the arm pauses at every chunk |
| SmolVLA via vla.cpp | 390 ms; 238 ms with flash attention | the engine was the 4× lever |
| ACT (PyTorch) | 59 ms per 100-action chunk | cheap; no language |
| VLA-JEPA 2B | 355 ms | world model off at inference |
| W1 surprise check (two views) | 17.6 ms; 19.1 ms with vla-server on the same GPU | fits a 33 ms tick |
| W1 25-step imagination | 153 ms | must run off the control thread |
| vla-server chunk while the monitor runs | 304 ms (from 238) | shared GPU contention |
| V-JEPA 2.1-G encoder, two views | 526 ms | off the robot |
| REKI 2.4B | unmeasured (G12 void) | the port would need the Qwen3.5 GDN stack in ggml |
| Control tick | 33.3 ms (30 Hz) | bus-limited |

What "chunk" and "tick" mean together: the policy produces H actions per inference (50 for SmolVLA); the controller consumes one per tick; a 238 ms inference covers ~7 ticks of latency; if the controller waits for the chunk (synchronous), it is blocked 14 % of the time at 26.6 Hz; if it does not (asynchronous), the elapsed actions are skipped and the new chunk's first executed action is ~7 steps in, which is where the boundary jump comes from and what prefix overwrite fixes.

Thermal: the GPU clock held at 1 162–1 172 MHz with Tj ≤ 83 °C under the vla.cpp sweep. Not throttling. Worth re-checking on long robot days.

Why no training on the Orin: 16 GB shared, slow memory bandwidth, ARM wheels for torch that you had to build yourself (5.5 h). Training rents an x86 GPU.

## 10.4 The Mac and the rented GPUs

The Mac (Apple silicon, MPS) is for offline evaluation and debugging: G1, G2, G6, G9c, G10 all ran there for free. MPS has gaps (RoboJEPA's predictor attention fails on MPS; CPU fallback is 10–23 s per step), so "runs on the Mac" is checked, not assumed.

vast.ai rents consumer and datacentre GPUs by the hour. Rules learned: rank offers by whole-job cost (GPU + disk + bandwidth), not hourly rate; interruptible bids cost more than they save on long runs (VLA-JEPA: $9.7 vs a $3.5–4.7 estimate); debug on the Mac, rent only for the run; every launch has a hard deadline (`bootstrap.sh`) and a local destroy timer so a hung instance cannot bill overnight; nothing that a protocol depends on stays only on the instance (W2, lost). Typical costs in this repo: a G0-sized world-model run $3–4 on an RTX 3090; a SmolVLA fine-tune $3; a 27B VLM inference pass $1–3 on an A100; the TurboVLA faithful run $2.15 on an RTX PRO 4000 at batch 64.

## 10.5 The datasets, physically

A LeRobot v3 dataset is: Parquet files holding many episodes each, one row per frame (`timestamp`, `frame_index`, `episode_index`, `observation.state` [6], `action` [6], `task_index`, and index columns); MP4 videos per camera with many episodes concatenated, each episode a time range inside the file; and metadata (`meta/info.json` with feature shapes and fps, `meta/episodes/…parquet` with one row per episode, `meta/tasks.parquet`, `meta/stats.json` with per-feature mean/std/min/max and quantiles for normalisation). The `episodes.jsonl` / `tasks.jsonl` files belong to the older v2.1 format. Chapter 10b covers the format and the pipeline in full. Loading a frame decodes the video at that timestamp (torchcodec; NVDEC on the Jetson, which is why ffmpeg had to be built with `--enable-nvdec`). `delta_timestamps` let a policy request a window of past observations and future actions around each frame; `action_is_pad` marks where the window ran past the episode end.

Your sets: `so101_bowls_all_trimmed` (182 episodes, 61 462 frames, idle starts trimmed), the cube/tidy set (~180), the 21-task merge (653), the 69 policy rollouts (recorded through the same recorder so they are datasets too, with the outcome label in a side CSV), the LEGO first session (60). Normalisation statistics are per dataset; a policy fine-tuned on one set and evaluated with another's stats silently misbehaves, which is one of the "ruled out" items in every rollout diagnosis.

## 10.6 What this hardware can and cannot show

Can: anything about tabletop pick-and-place with two cameras; anything about evaluation, judging, monitoring, data curation; chunk-boundary and inference-engine work; world-model work where the question is about object state, not arm state; small-data imitation learning.

Cannot: force or tactile methods (no sensing); bimanual coordination (one arm; a second SO-101 would change this); humanoid or locomotion anything; large-scale pretraining (compute); methods whose value only appears with hardware-synchronised multi-camera rigs or a zenithal view; anything needing a simulator you have not built.

The honest sentence for an interview: "I can test claims about perception, evaluation, data and execution on a $300 position-controlled arm with two webcams and a Jetson; I cannot test claims that need force, two arms, or scale, and I say so in every write-up."

## 10.7 Exercises

1. Read a servo's present position, goal position and (if exposed) current from the bus with the LeRobot driver, at 30 Hz for 10 s, with the arm held by hand against a soft obstacle. Plot all three. Write what a contact detector could and could not get from this.
2. From the URDF, compute forward kinematics for the rest pose and one grasp pose by hand (or in 20 lines). Compare to the harness's `so101_kinematics.py`.
3. Open one episode of the bowls dataset with the LeRobot API; print the feature shapes, the fps, the stats; decode frame 100 of each camera; plot `action` vs `observation.state` for the shoulder pan over the episode and measure the tracking lag in frames.
4. Measure the Orin's SmolVLA chunk latency yourself through vla.cpp (20 reps, 3 warmups) and compare to 238 ms. If it differs by more than 10 %, find out why before trusting any other number.
