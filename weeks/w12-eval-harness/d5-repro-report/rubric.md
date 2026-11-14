# Rubric

Score each line 0 / 1 / 2.

| # | Criterion | Senior signal (1) | Staff signal (2) |
|---|---|---|---|
| 1 | Goals | "measure success rate" | decisions the program serves (research ranking, promotion, regression catching), per-task success criteria written in advance, non-goals |
| 2 | Tiers | sim + real | cheap → expensive funnel: offline metrics and world-model/sim screen only to prune, real robots decide; each tier's job and its known blind spots |
| 3 | Protocol | fixed tasks | fixed start configurations per task, blocked counterbalanced schedule across stations, blinding, reset procedures, calibration arm (frozen reference policy every session) |
| 4 | Statistics | CIs | paired comparisons, Holm/FDR across tasks and candidates, MDE per task stated, sequential or two-stage designs to save robot time, seeds vs rollouts |
| 5 | Budget | rollouts per policy | robot-hours per release derived (rollout time + reset × rollouts × candidates × tasks), who operates, how many candidates reach real eval |
| 6 | Station drift | "monitor robots" | A/A checks with the reference policy per station, station as a factor in the analysis, automatic quarantine when the reference drops, hardware health logs |
| 7 | Labels | operators decide | written criteria, failure taxonomy, double labelling of a sample with κ, video retained, automatic stage detection where possible |
| 8 | Proxy validation | "use the simulator" | sim/world-model scores validated by rank correlation (and MMRV) against real outcomes on past checkpoints; re-validated when the model or task changes |
| 9 | Deployment realism | none | evaluation runs the field configuration: same compute, latency budget, execution mode (async/RTC), firmware; latency percentiles reported with success |
| 10 | Reporting & promotion | dashboard | manifests per run, generated tables, per-task and per-cause results, promotion rule pre-registered (non-inferiority margins per task, no regression in safety stops), staged fleet rollout with field monitoring |

0–9 senior or below · 10–14 senior+ · 15–17 staff · 18–20 strong staff.
