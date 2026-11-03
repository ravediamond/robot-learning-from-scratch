# Chapter 10b. The data pipeline: from teleoperation to a training batch

Every result in this book starts as a recorded episode, and most of the failures in chapter 8 started there too: a hand in the frame, a camera mounted upside down, aborted 11-frame episodes, an idle start the policy learned to copy, normalisation statistics from the wrong dataset. A research engineer is expected to own this pipeline end to end and to find those problems before a training run does. This chapter explains it from zero for the LeRobot format used throughout the book (version 3.0, checked against the `lerobot` 0.6.1 source on 2026-10-24).

## 10b.1 What a robot dataset is

- **An episode** is one attempt at a task, from start to stop: a few seconds to a minute.
- **A frame** is one time step of an episode, at a fixed rate (**fps**, here 30): one row of numbers plus one image per camera.
- **Features** are the named quantities stored per frame. The LeRobot convention:
  - `observation.state`: the robot's own joint positions (6 numbers for the SO-101: five arm joints and the gripper);
  - `observation.images.<camera>`: one image per camera (here `front_top` and `wrist`);
  - `action`: what was commanded at that frame (6 numbers; in teleoperation, the leader arm's joint angles);
  - bookkeeping: `timestamp` (seconds since the episode start), `frame_index` (within the episode), `episode_index`, `index` (global row number), `task_index`.
- **The task** is a natural-language string per episode ("pick up the blue bowl and put it in the pink bowl"), stored once and referenced by `task_index`. Language-conditioned policies read it; it must be a full, consistent sentence (a telegraphic "Pick colored block" is out of distribution for a pretrained language model).

A training example is not a single frame: a chunked policy needs the current observation (perhaps a few past ones) and the next H actions. 10b.4 shows how the loader builds those windows.

## 10b.2 The format on disk (LeRobot v3.0)

```
meta/info.json                                    # the schema: version, fps, features, counts, path patterns
meta/stats.json                                   # per-feature statistics for normalisation
meta/tasks.parquet                                # task strings and their indices
meta/episodes/chunk-000/file-000.parquet          # one row per episode: length, tasks, where its rows and video segments are
data/chunk-000/file-000.parquet                   # the frame table, many episodes per file
videos/<camera>/chunk-000/file-000.mp4            # the video, many episodes concatenated per file
```

- **`info.json`** holds `codebase_version` ("v3.0"), `fps`, `robot_type`, `total_episodes`, `total_frames`, the path templates (`data_path`, `video_path`), and `features`: for each feature its `dtype`, `shape` and `names` (for `observation.state`, the six joint names; for video features, height, width, channels and codec information). Read it first when you meet any dataset.
- **Frames live in Parquet** (a compressed columnar table format), many episodes per file; a file grows to about 100 MB before a new one starts, and a chunk directory holds up to 1 000 files.
- **Video lives in MP4**, many episodes concatenated into one file per camera (up to about 200 MB each). An episode is a **time range** inside that file, recorded in its row of `meta/episodes` (`videos/<camera>/from_timestamp`, `to_timestamp`). Loading a frame means decoding the video at a timestamp, not reading an image file.
- **`meta/episodes`** has one row per episode: its `length`, its tasks, its global row range (`dataset_from_index`, `dataset_to_index`), which data and video files it is in, and per-episode statistics.
- **`stats.json`** holds, per feature, `mean`, `std`, `min`, `max`, a `count`, and quantiles (`q01`, `q10`, `q50`, `q90`, `q99`). Policies normalise inputs and outputs with these.

**Older versions.** Datasets recorded before v3.0 (v2.1) store one Parquet and one MP4 *per episode* and keep metadata in JSON-lines files (`meta/episodes.jsonl`, `meta/tasks.jsonl`, `meta/episodes_stats.jsonl`). LeRobot ships a converter (`convert_dataset_v21_to_v30.py`). When a script fails on a community dataset, check `codebase_version` first.

**Size.** In this repository, 8 rollout episodes of two 640×480 AV1-encoded cameras at 30 fps took 238 MB for 5 787 frames, about 41 kB per frame including both images (`shared/gate/data/ROLLOUT_CATALOG.md`). Video dominates; the Parquet table is small. At that rate the 61 462-frame bowls set is roughly 2.5 GB, and the 600-episode data engine of the use-case brief (about 25 s per episode) roughly 18 GB. Budget disk and upload time accordingly.

## 10b.3 Recording

`lerobot-record` connects the follower arm, the leader (or a policy) and the cameras, and writes one episode at a time:

```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.cameras='{"front_top": {...}, "wrist": {...}}' \
  --teleop.type=so101_leader \
  --dataset.repo_id=${HF_USER}/my_task \
  --dataset.single_task="pick up the blue cup" \
  --dataset.num_episodes=25 \
  --dataset.fps=30
```

Each control tick it reads the follower's joints and the cameras, reads the leader's joints as the action, writes the goal to the follower, and appends a frame. Between episodes there is a reset period. Video is encoded when the episode is saved.

**The recording checklist** (from `shared/SO101_RECORDING_GUIDE.md`, written after a 375-episode dataset trained badly; every item is a failure that happened):

1. **Nothing in frame at recording that will be absent at deployment.** The worst defect found: the operator's hand visible at the edge of the top camera through entire episodes. A signal present in 100 % of training data and 0 % of deployment is a perfect spurious cue.
2. **Mount cameras rigidly and right side up**, check the live preview, and do not touch them between sessions (an inverted camera was found only by reading inverted text in a frame).
3. **Trim idle time.** Start at the reach, stop after the goal is reached. A 1.6 s idle start in early recordings taught a policy to freeze at the beginning of every episode.
4. **Slow down on fast transit moves.** Motion blur appeared in short bursts during the fastest segments only, and was first misdiagnosed as "always blurry" from two arbitrary frames.
5. **Delete false starts.** Five episodes of exactly 11 frames (0.37 s) were aborted recordings mixed into training.
6. **One full, grammatical task string per task**, identical across its episodes.
7. **Lock camera exposure and white balance**, control the light, and log the setting.
8. **Record a few more episodes than you need** and discard bad ones by a written rule, not by feel.

## 10b.4 Loading: windows, padding and video decoding

`LeRobotDataset("<user>/<name>")` downloads (or reads from the local cache) and behaves like a PyTorch `Dataset` (P2.8): `ds[i]` returns a dict of tensors for global frame i, decoding the camera images at that frame's timestamp.

**`delta_timestamps`** asks for a window around each frame, in seconds relative to it. The loader converts seconds to frame offsets with `round(seconds × fps)`, gathers those frames **from the same episode**, and returns a boolean `<key>_is_pad` for offsets that fall outside the episode (it repeats the edge frame in their place). The loss must ignore the padded positions (P2.6). The same logic in a few lines:

```python
import torch

fps = 30
delta_timestamps = {
    'observation.state': [-1 / fps, 0.0],                     # the previous and the current frame
    'action': [i / fps for i in range(10)],                   # the next 10 actions: a chunk of 10
}
delta_indices = {k: [round(d * fps) for d in v] for k, v in delta_timestamps.items()}
print(delta_indices)

# one episode of 12 frames, 6 joints; values = frame index so the result is readable
ep_from, ep_to = 100, 112                                     # global row range of this episode
action = torch.arange(ep_from, ep_to).float()[:, None].repeat(1, 6)

def window(global_idx, key_offsets, column):
    idx = torch.tensor([global_idx + o for o in key_offsets])
    is_pad = (idx < ep_from) | (idx >= ep_to)                 # outside this episode
    idx = idx.clamp(ep_from, ep_to - 1)                       # repeat the edge frame instead
    return column[idx - ep_from], is_pad

a, pad = window(107, delta_indices['action'], action)         # a frame 5 from the end
print(a[:, 0].tolist())
print(pad.tolist())
```

Output: offsets `[-1, 0]` for the state and `[0 … 9]` for the actions; for a frame five steps before the end of a 12-frame episode, actions 107–111 are real and the last five are copies of 111 marked `True` in the pad mask. If the loss does not use the mask, the policy learns to stop moving at the end of every episode.

**Video decoding** is the expensive part of loading: each sample decodes one frame per camera from compressed video (torchcodec or PyAV; on the Jetson, with the hardware decoder). Use several DataLoader workers, and expect data loading, not the GPU, to be the bottleneck for small models.

**Image transforms** (augmentation: colour jitter, small crops, camera dropout) are applied at load time in training only (P1.7). Keep them off for validation and evaluation.

## 10b.5 Splits, statistics and normalisation

- **Split by episode** (P1.7, P2.8): choose episode indices for training and validation with a fixed seed, and load each split with the `episodes=[…]` argument. Frames of one episode never appear on both sides.
- **Normalisation statistics** come from the training data and travel with the checkpoint. A policy normalises its inputs and un-normalises its outputs with them at inference. Evaluating a policy with another dataset's statistics (or recomputed ones) changes every input silently; it is on the "rule out first" list of every rollout diagnosis in chapter 8.
- **Mean/std vs quantiles.** Mean and standard deviation are sensitive to a few extreme values (a gripper that snaps fully open once); quantile-based normalisation (mapping q01–q99 to −1…1) is more robust. Know which one your policy uses.
- **Merging datasets** (several tasks into one training set) re-computes statistics over the union and re-indexes episodes and tasks. Check `total_episodes`, the task list and the statistics after every merge.

## 10b.6 Audit before training

Ten minutes that save a training run. Statistics alone do not reveal the problems that matter: the bad 375-episode dataset had clean action and state statistics; its defects were in the footage.

1. **Counts.** `total_frames` equals the sum of episode lengths; every camera's decoded frame count equals the episode's frame count (the rollout catalog checks this for every dataset).
2. **Episode lengths.** Histogram them. Very short episodes are false starts; very long ones are usually operator trouble.
3. **Look at frames.** For a sample of episodes spread across the dataset, decode the first, middle and last frame of each camera and look at them: hands, mounts, blur, lighting, occlusion. Sample systematically across whole episodes, not two arbitrary timestamps.
4. **Idle segments.** Plot joint speed over time for a few episodes; long zero-speed stretches at the start or end should be trimmed.
5. **Action vs state.** In teleoperation the action (leader) should lead the state (follower) by a small, consistent lag. Large or variable offsets mean calibration mismatch or bus trouble.
6. **Ranges.** Per-joint min and max inside the calibrated range; no constant dimension (a disconnected servo reads constant).
7. **Tasks.** One string per intended task; no typos creating extra tasks.
8. **Labels.** Outcome labels (success, failure class) for rollouts live in a side table keyed by episode index; check the join covers every episode.

## 10b.7 Rollouts are data too

Every policy run on the robot should be recorded through the same recorder, so it becomes a dataset with the same schema: the policy's own actions, both cameras, the joint states, plus a side table with the outcome, the failure class and the context (policy checkpoint hash, engine build, layout identifier, verifier verdict and tier). This is the data the world-model, judge and curation experiments of Part II need, and the reason the data-engine use case exists.

Rules that keep it usable: **never imitate failed rollouts** (they go to judges, critics and world models only; AGENTS.md §1); label outcomes with the most credible verifier available and record which one (chapter 13's tiers); keep evaluation layouts out of training data; record the context fields at recording time, because they cannot be recovered later.

## 10b.8 Storage, versions and provenance

- **The Hugging Face Hub** hosts datasets (public or private) with git-like **revisions**; `push_to_hub` uploads, and loading can pin a revision or tag so that a result can be reproduced with exactly the data it used.
- **Never commit datasets to a code repository.** Keep a catalog file (what exists, where, how many episodes and frames, which labels) and checksums of anything a protocol depends on.
- **Nothing a result depends on lives only on a rented machine.** A world-model checkpoint that existed only on a destroyed GPU instance (W2) had to be retrained; datasets are the same.
- **Write down what was done to a dataset** (trimming, merging, relabelling) as a change log next to it; a relabelled episode changes every number computed on it (the label-change file in `data/labels/`).

## 10b.9 Exercises

1. Open `meta/info.json` of one of your datasets and write, from memory first, the features, shapes, fps and episode count; then check.
2. Implement the 10b.6 audit as one script that prints a report for any LeRobot v3 repository id; run it on the bowls set and the 69 rollouts; list every finding.
3. Load a dataset with `delta_timestamps` for a 50-action chunk and confirm, on the last frame of an episode, that 49 actions are padded and that your training loss ignores them.
4. Compute normalisation statistics on a training split of episodes only, then on all episodes; report the largest per-joint difference and say whether it would matter for a policy.
