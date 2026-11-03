# Weekly reading slot (20 min, once a week, no code written)

The labs train small, clean exercises. Real code is 400-line files you did not write: LeRobot,
your harness, what Claude Code produced on the Orin. Reading them fast and correctly is its own skill,
and it is the one you need to check Claude Code's work. One real file per week.

## Protocol
1. Timer: 20 minutes. Read only. No running, no editing, no AI summary.
2. Fill `TEMPLATE.md` into `notes/wN.md` (10 lines max).
3. Then (not before) compare with the map: `knowledge/09-lerobot-reading-map.md` for LeRobot files,
   `knowledge/10-orin-harness-audit-map.md` for your harness. Mark what you got wrong.
4. Log the time and the misses in `PROGRESS.md` (Reading row of the week).

Versions: LeRobot `0.6.2` at commit `06f78b6`; road-to-robotics at `c2a37e5`. Get them once:
```bash
git clone https://github.com/huggingface/lerobot ~/src/lerobot && git -C ~/src/lerobot checkout 06f78b6
git clone https://github.com/ravediamond/road-to-robotics ~/src/road-to-robotics   # or your existing clone
```
If your Orin pins another LeRobot version, read that one too and note the differences: that is the point.

## Targets (paths relative to the repo root; LeRobot paths under `src/lerobot/`)
| Week | File | Read for | Matches |
|---|---|---|---|
| 1 | LeRobot `processor/normalize_processor.py` lines 269–430 | where and how inputs/actions are normalised, which stats, inverse path | W1 D5 |
| 2 | LeRobot `datasets/dataset_reader.py` (search `delta_window`) + `datasets/factory.py: resolve_delta_timestamps` | how a (obs, action chunk, `action_is_pad`) sample is built | W2 D1, D4 |
| 3 | LeRobot `policies/smolvla/smolvlm_with_expert.py` (570 lines) | token blocks, which VLM layers run, attention masks, shapes | W3, W5 D2 |
| 4 | LeRobot `policies/smolvla/modeling_smolvla.py` lines 741–863 | the flow loss and sampler: which end is noise? how many steps? | W4 D3 |
| 5 | LeRobot `scripts/lerobot_train.py` lines 181–330 and 529–760 | seed, processors, optimizer preset, resume, one update step | W2 D2, W9 D1 |
| 6 | road-to-robotics `journal/papers/failure-monitors/length_analysis.py` (127 lines) | your published eval: unit of resampling, strata, ties | W6 D2–D3 |
| 7 | road-to-robotics `projects/vla-jepa-world-model/wm_train/train_lewm.py` (239 lines) | your real LeWM vs your mini LeWM: losses, latent size, what is logged | W7 D4 |
| 8 | road-to-robotics `agent/policies/rtc_controller.py` (274 lines) + LeRobot `policies/rtc/action_queue.py` | stale-action skipping, boundaries, timing | W8 D3 |
| 9 | road-to-robotics `scripts/ws.py` (`cmd_verify`, line 741) + `.claude/hooks/pre_bash.py`, then the audit routine of `knowledge/10` §5 on one archived item | what the harness guarantees and what it does not | W8 D4–D5 |
