# Model answer: evaluation program for a generalist manipulation policy

**Decisions served.** (1) Rank research candidates every two weeks; (2) promote one checkpoint to the
fleet with no regression; (3) catch regressions early. Non-goals v1: absolute "generalisation scores",
evaluation of every intermediate checkpoint on real robots.

**Tiers.**
1. Offline (minutes, free): validation losses, action error on held-out episodes, failure-detector
   scores. Only sanity: these rank policies badly (W6 D4).
2. Proxy (hours, cheap): simulator for the 2 covered tasks, world-model rollouts for all 5. Used to
   PRUNE 5 candidates to 2, only on tasks where its ranking agreed with real outcomes on past releases
   (Spearman ≥ 0.8, MMRV low, rechecked every release).
3. Real robots (days, expensive): the decision. 2 candidates + current production + a frozen reference.

**Protocol.** Per task, 10 start configurations marked on every station, identical object sets.
Blocked schedule: each block = one start, all four policies in rotating order, spread over stations
and both sites, operators blind to aliases. Written success criteria and failure taxonomy per task,
stage detection from proprioception, video for every rollout.

**Budget.** About 1.5 min per rollout including reset. 4 policies × 5 tasks × 40 rollouts = 800
rollouts ≈ 20 robot-hours: 12 stations × 2 hours = one morning. MDE at 40 rollouts ≈ 30 points per task
unpaired; pairing and pooling across tasks help. For the promotion decision use a two-stage design:
stage 1 at 20 rollouts per task, stop candidates clearly worse; stage 2 adds 40 for the survivors.

**Statistics.** Per task: paired McNemar vs production with Holm across tasks; pooled across tasks with a
mixed model (task and station as factors). Promotion rule, pre-registered: non-inferiority on every
task with a margin of 10 points (one-sided), superiority on the pooled metric, no increase in safety
stops (zero tolerance, any increase blocks).

**Drift.** The frozen reference runs in every session on every station (A/A). Its success per station
is charted; a station falling outside its control limits is quarantined and its rollouts re-run.
Hardware logs (servo temperature, calibration error, camera exposure) in each manifest.

**Labels.** Operators label with the taxonomy; 15% of rollouts relabelled from video by a second person
blind to the alias; κ per task reported; low κ triggers a definitions fix.

**Deployment realism.** Evaluate on the fleet's compute module with the field execution mode (async
with RTC, same chunk settings), same firmware. Latency p50/p95/p99 and deadline misses reported with
success; a candidate that misses the latency budget is not promotable whatever its success.

**Infrastructure.** A scheduler that generates blocked run sheets per station; a launcher that loads
weights by fingerprint and refuses mismatches; schema-validated logs; manifests per session;
generated reports (tables, per-cause rates with CIs, funnels, videos of failures); a register of every
release's results.

**After promotion.** Staged fleet rollout (10% of robots, then all) with field success and intervention
rates monitored against the evaluation's prediction; disagreements feed the next protocol revision.

**Not in v1.** Automatic failure tagging from video (start manual, measure agreement, then automate);
fully automatic resets; continuous real-robot evaluation of every checkpoint.
