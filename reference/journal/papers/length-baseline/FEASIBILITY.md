# Feasibility — "Episode length as the null model for robot reward models" (short paper)

WS-073, 2026-10-31. Memo only: nothing was run. Depends on WS-070 (`journal/papers/failure-monitors/FEASIBILITY.md`),
whose §2 table is the evidence base; this memo repeats only what the short paper needs.

**Which paper this is.** If WS-070 is a go, this is **its §3–4, not a separate paper**. Write it separately only if
the owner wants a 4-page note out first (it needs only $0 analyses) and the 8-page monitors paper later; the two
cannot both claim the length result as new.

## 1. Claim in one sentence

On every SO-101 rollout set we have, episode length is a failure detector that a learned reward model must beat and
did not: 0.938 vs Robometer-4B 0.818 on 2,099 ArmnetBench rollouts, with Robometer's AUROC inside length quintiles at
0.687 (bar 0.75) and a CV gain of +0.016 (bar 0.03); and reward-model papers should report a "beyond-length" statistic.
True if the within-policy and fixed-time-truncation analyses keep length at or above Robometer.

## 2. Evidence in hand

| gate | n | length | learned / judge | anchor |
|---|---|---|---|---|
| G9a-AB complete | 2,099 (7 policies, ArmnetBench) | **0.938** | Robometer 0.818 [0.790, 0.843]; quintile 0.687; CV gain +0.016 | `robometer/results/g9a_ab/report.md` |
| G9a-AB partial (superseded) | 1,637 | 0.934 | 0.861; quintile 0.776; CV gain +0.019 | same |
| G9a-own | 88 own | **0.52** (the counter-example) | Robometer 0.577; policy identity 0.878 | `g9a_own/report.md` |
| REAP monitors | 50 own | elapsed time 0/8 | risk head (v1) 6/8 at 7/42 → v2 3/8 at 5/42 | `2026-10-22_reap` |
| W1-A | 50 own | — | world-model surprise 2/8 | `2026-10-19_w1_screening` |
| Judge τ | 50 own | — | Gemini windows 0/8 | `2026-10-19_judge_tau` |
| G-flora / -abl | 50 own | — | Flora 8/8 at 2/42; Qwen3.5-27B 7/8 at 2/42 (both beat length on own data) | `2026-10-20_g_flora*` |

Within-task quintiles on ArmnetBench (already in the report): median 0.762, pair-weighted 0.800 (complete set). The
own-rollout counter-example has a cause the paper must state: our operators stopped episodes by hand, ArmnetBench
episodes run to a timeout.

## 3. Gaps, costed

| gap | resource | queued? |
|---|---|---|
| within-policy length AUROC on ArmnetBench (7 policies) | public metadata, ~1 h CPU, $0 | new |
| fixed-time truncation: AUROC of every detector on episodes cut at a common horizon (removes the timeout signal) | length: $0; Robometer: needs its per-episode scores — on the old Mac (`ab_top.jsonl`, `ab_top_rescore.jsonl`) or a ~$6 re-score | new |
| per-task beyond-length statistic (quintile AUROC, CV gain) as a reusable script | ~2 h, $0 | new |
| one more learned reward model (e.g. SARM or TOPReward) through the same statistic | ~$5–10 GPU | new |
| **total** | **≈ $0–16, ~1 day, no arm** | |

## 4. Related work

Robometer 2603.02115 (cites "trajectory length as a proxy for quality" as a weakness of prior data, reports no length
baseline); SARM 2509.25358 (dismisses "simple proxy heuristics like task duration"; no length baseline); TOPReward
2602.19313, STDR 2606.31377, AXIS-Bench 2610.18869, GVL 2411.04549 (no length mentioned); VLLR 2604.00055, SARM2
2606.10305 (use episode length inside a reward or metric, not as a baseline); *No Free Checker* 2609.09250 (verifier
survey; no length baseline); Mulligan 2610.05882 (duration as an efficiency metric). All read in the mirror by full-text
search on 2026-10-31. Not on disk: RoboValue 2610.26663, Flora-Rewarder (NVIDIA), and the offline-RL "timeout" literature
(Pardo et al. 2018, *Time limits in reinforcement learning*) — named, not read here.

## 5. Objections

**First: trivial — failed episodes hit the timeout, so length is the label.** That is the paper's job, not an
objection to it: the within-policy and truncated analyses (§3) separate "length is informative" from "length is the
timeout". Our own data (0.52) shows the other regime. **Second: were the judges given a fair shot?** Robometer was
run as its authors ship it, on two camera views, with a calibrated threshold (t* = 0.124), and the input-format result
(G-flora-abl) shows how much the input matters; cite it. Two task families only.

## 6. Venue

Primary: a 2027 workshop on robot evaluation (RSS / CoRL; to confirm), 4 pages. Fallback: arXiv-only technical note.
Endorsement: none named (no contacts list in the repo).

## 7. Minimal vs full paper

This is the minimal version (4 pages, one table, one figure). The full version is WS-070.

## 8. Risk of a null result

If within-policy length is weak, the note becomes "evaluate reward models within policy, or the timeout does the
work" — still the same recommendation, still 4 pages.

## 9. Recommendation

**go-after-X**, X = the two $0 length analyses (~2 h). Then write it **as WS-070's §3–4**; a standalone note only on
the owner's call. Earliest: 4-page draft one week after X.

## 10. Skeleton

Titles: (1) *Episode length is the reward model to beat*; (2) *A null model for robot reward models*; (3) *Beyond
length: reporting what a reward model adds*. Abstract (≤ 150 words): Reward models for robot policies are evaluated
by how well they separate failed from successful rollouts. On 2,099 public SO-101 rollouts, episode length alone gives
AUROC 0.938, above Robometer-4B zero-shot (0.818); inside length quintiles Robometer reaches 0.687 [to run: within-policy,
fixed-time]. On 88 own rollouts stopped by an operator, length is at chance (0.52), and only judges given the right
input beat it. We propose a beyond-length statistic every reward-model paper should report. Sections: Intro · Data ·
The length baseline · Beyond length · When length fails · Recommendation. Figures: (1) length distributions by outcome
`[to run]`, $0; (2) AUROC per detector inside length quintiles — `g9a_ab/report.md` (today).
