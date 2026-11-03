# Feasibility — "Elapsed time is the reward model to beat": failure monitors and judges on low-cost VLAs

WS-070, 2026-10-31. Memo only: nothing was run. Sources: `journal/EXPERIMENTS.md` rows quoted by name,
`shared/gate/data/` READMEs, `projects/vla-jepa-world-model/robometer/results/g9a_ab/`, the register, the papers
mirror (`papers-sync` 2026-10-31).

**Correction to the prompt's framing, from the rows.** Two of the prompt's statements do not hold. (1) "Length alone
beats … Flora-Rewarder": on our own rollouts Flora-Rewarder-27B **passed** (8/8 failures at 2/42 successes, AUROC 1.00,
G-flora), and length is near chance there (AUROC 0.52, G9a-own; elapsed time 0/8, REAP monitors). (2) "The risk head
is the only monitor that beats length (AUROC 0.93, onset check pending)": REAP v2 (pre-registered `6acb9d3`) removed
time and length and the 0.93 fell to 0.79; **no monitor met the bar** and v2 supersedes v1. The memo is built on the
rows as they stand.

## 1. Claim in one sentence

Episode length is a failure detector that learned zero-shot judges must beat and on the public SO-101 benchmark do not
(length AUROC 0.938 vs Robometer-4B 0.818 on 2,099 ArmnetBench rollouts; beyond-length quintile median 0.687 < 0.75);
on our own rollouts, where length is uninformative (0.52), only judges given the right input — a robot-trained or
27B-scale VLM on a 64-frame two-view clip, or a deterministic top-camera check at rest — catch failures, and every
mid-episode monitor tested falls short of 5/8 at ≤ 1/20 false interventions. True if the within-policy and
fixed-time-truncation analyses of §3 keep length ahead of Robometer on ArmnetBench.

## 2. Evidence in hand

| gate | date | n | result | status | anchor |
|---|---|---|---|---|---|
| G9a-AB complete | 10-16 | 2,099 rollouts, 7 policies | Robometer-4B top view AUROC 0.818 [0.790, 0.843]; **length 0.938**; beyond-length quintile median 0.687 (bar 0.75), CV gain +0.016 (bar 0.03) | fail (not a zero-shot judge) | EXPERIMENTS; `robometer/results/g9a_ab/report.md` |
| G9a-AB (partial, superseded) | 10-15 | 1,637 scored | 0.861; length 0.934; quintile 0.776 | qualified pass, superseded | same |
| G9a-own | 10-16 | 88 own (64 / 24) | Robometer 0.577 [0.44, 0.70]; **length 0.52**; policy identity 0.878 | fail (baseline) | `g9a_own/report.md` |
| G9c-3 deterministic judge (at rest) | 10-16 | 88 | 94.3 % accuracy, 0 FP (FP 0/25 [0, 13.3 %], FN 5/62) | narrow fail / operational pass | `judge/results/g9c3/report.md`; Verifier cards |
| Judge τ (Gemini sliding windows) | 10-19 | 50 (42 / 8) | caught **0/8** at 1/42 | judge not usable | `2026-10-19_judge_tau` |
| G16-B (Gemini checkpoints) | 10-19 | 50 | failures 8/8 but successes 41/42 flagged | fail | `2026-10-19_g16b` |
| W1-A (world-model surprise, drift, reflex) | 10-19 | 50 | L2 surprise **2/8** at 2/42 (AUROC 0.86); drift 0/8; reflex flags 42/42 | fail | `2026-10-19_w1_screening` |
| W1-C2 (FAIL-Detect / SAFE band on W1) | 10-19 | 22/22 split × 20 seeds | successes 22.7 % flagged, failures 46 % | fail | EXPERIMENTS |
| G-flora | 10-20 | 50 | **8/8 at 2/42, AUROC 1.00** | pass | `2026-10-20_g_flora` |
| G-flora-abl | 10-20 | 50 | Qwen3.5-27B 7/8 at 2/42 (pass); 9B 16/42 FP (fail); Gemini 3.8 Flash 8/8 at 3/42 (fail by one) | input format did the work | `2026-10-20_g_flora_abl` |
| REAP monitors (v1) | 10-23 | 50 (+69) | risk head 6/8 at 7/42 (AUROC 0.93); surprise 2/8; **elapsed 0/8** | not proven → superseded | `2026-10-22_reap` |
| REAP v2 | 10-23 | 50 (+69) | logistic 3/8 at 5/42 (0.789); MLP 6/8 at 13/42; action-conditioned (+0.07–0.08 over shuffled) | **fail** (no monitor at the bar) | same, v2 |
| PRIME | 10-29 | — | graph check failed (held-out Spearman 0.51) | void | `2026-10-23_prime` |
| REMORA readout | 10-31 | 34 test | flips 0.70 × argmax; setback recall 2/7 | P2, P3 fail | `projects/remora-readout` |

Pre-registered (commit before the run): G-flora (`1c50a5d`), G-flora-abl (`80c0fe8`), W1-A (`37497f0`), G16-B
(`aeb99e5`), REAP v1 (`2083fb0`), v2 (`6acb9d3`), PRIME, REMORA. G9a-AB had a protocol (`G9A_AB_PROTOCOL.md`); the
"complete set" rule was an owner decision after the partial run (stated in the row). Labels: operator labels for our
88 / 50 episodes, not blind to policy; ArmnetBench labels are the dataset's. Statistic: bootstrap CIs on the AUROCs;
the per-gate counts (k/8 at j/42) carry no CI — Wilson intervals are recomputable from the counts ($0).

## 3. Gaps, costed

| gap | resource | queued? |
|---|---|---|
| **within-policy length AUROC and fixed-time truncation on ArmnetBench** (the central objection) — needs only episode lengths, labels and policy ids, all in the public dataset metadata | ~2 h CPU, $0 | new |
| the same two analyses for Robometer's scores — the per-episode score files (`ab_top.jsonl`, `ab_top_rescore.jsonl`) are on the old Mac archive, not on the Orin | recover from the Mac, or re-score on vast (~$6) | new |
| Wilson / bootstrap CIs on every k/8-at-j/42 count | $0, 1 h | new |
| a second own task family (today bowls; cubes data exists) for the judges | ~1 arm day + Flora / Qwen calls (~$3) | new |
| a commercial-licence judge (Qwen3.5-27B fine-tune or prompt) on fresh rollouts | ~$5 vast + 1 arm day | the G-flora write-up proposed it |
| blind relabel of the 50-episode set by a second person | ~2 owner hours | new |
| REMORA / PRIME follow-ups | none needed for this paper (both reported as is) | WS-065 done |
| **total** | **≈ 2 arm days, ~$10–15, ~6 hours** | |

## 4. Related work

| paper | what it showed | what differs here |
|---|---|---|
| 2603.02115 Robometer | general reward model; failure-detection F1 0.81 zero-shot (Table IV) | on ArmnetBench it falls below episode length |
| 2606.08881 SO-101 failure / recovery benchmark | failure taxonomy on the same arm | taxonomy, not a test of judges against a null model |
| 2609.09250 *No Free Checker* (survey of ~150 verifiers) | availability and credibility of verifiers trade off | a survey; we measure the trade-off on one arm |
| 2609.11445 FARM | supervised readout on frozen VLA-JEPA predictive states for online failure detection | closest to REAP v2; we find the action-conditioned head below the bar on our data |
| 2609.36808 Spotter | a parallel VLM screener that intervenes on detected failures | our Gemini window and checkpoint judges (0/8; 41/42 FP) are the low-cost version of that screener |
| 2610.39125 REMORA | error-aware progress readout | WS-065: steadier, but no setback detection on our evidence |
| 2610.38508 SaiVLA-JEPA | JEPA conditioning helps control; future prediction needs care | control side; a null on the monitor side is consistent |
| 2509.25358 SARM, 2602.19313 TOPReward, 2604.00055 VLLR | progress / reward models from stages, token probabilities, VLM progress | none reports a length or elapsed-time baseline (to confirm per paper, §4 of WS-073) |
| ReAP 2610.16309, SAFE 2506.09937, FAIL-Detect 2503.08558, VLA-Corrector 2609.06508, PRIME 2610.23769, VeriVLA 2610.21985, RoboValue 2610.26663 | — | **not read here** (not in the mirror) |

(The prompt gives 2609.11445 as ReAP; in the mirror that id is FARM. ReAP's id in the register is 2610.16309.)

## 5. Objections

**First: length is trivially predictive because failed episodes run to the timeout.** On ArmnetBench, partly yes,
and that is the point: a reward model that cannot beat the timeout is not adding information. The answer the paper
needs is the within-policy and fixed-time-truncation analyses (§3, $0 for length). On our own rollouts the operator
stopped episodes by hand, so length carries no signal (0.52) — the two datasets bracket the effect.

**Second: the judges were not given a fair shot.** The flora input-format post is the answer: the same 64-frame
two-view clip with NVIDIA's prompt makes Qwen3.5-27B pass and Gemini miss by one false positive; the judge failures
are input failures as much as model failures. Also: the risk head is trained and tested on the same 50 episodes
(5-fold by episode, so episodes do not leak, but the task and the policies do); two task families only; labels not
blind for the early gates.

## 6. Venue

Primary: a 2027 workshop on robot evaluation / failure detection (RSS or CoRL, deadlines to confirm when announced).
Fallback: RA-L (rolling, 6 + 2 pages, single-blind). arXiv first: yes, once the length analyses of §3 are done.
Endorsement (cs.RO): no contacts list in the repo; none named.

## 7. Minimal vs full paper

4 pages = WS-073's short paper (length as the null model, ArmnetBench + own, one table, one figure). 8 pages: + the
judges on our rollouts (deterministic, Flora / Qwen / Gemini, input format), the mid-episode monitors (W1-A, REAP v2,
FAIL-Detect / SAFE band), REMORA and PRIME as reported nulls. Needs from §3: the length analyses and the CIs (4 p.);
the second task family and blind relabel (8 p.).

## 8. Risk of a null result

If within-policy length falls well below Robometer, the headline flips to "length is a dataset artefact of
timeouts; judges must be evaluated within policy" — still a methods note worth 4 pages. The judges-on-own-rollouts
half does not depend on it.

## 9. Recommendation

**go-after-X**, X = the length analyses on the public metadata ($0, ~2 h) and recovery of the Robometer score files.
Write the 8-page paper only after the second task family (~2 arm days at ~2 per week). Earliest: 4-page arXiv mid-
December 2026; 8-page late January 2027. **WS-073 (the short paper) is this paper's §4, not a separate paper**, unless
the owner wants only the 4-page version.

## 10. Skeleton

Titles: (1) *Elapsed time is the reward model to beat*; (2) *Judging robot rollouts on a $200 arm: null models,
input formats and mid-episode monitors*; (3) *What catches a failed rollout on low-cost hardware?*

Abstract (≤ 150 words): Learned reward models and VLM judges are proposed as failure detectors for robot policies. On
2,099 public SO-101 rollouts from 7 policies, episode length alone separates failures (AUROC 0.938) better than
Robometer-4B zero-shot (0.818), and Robometer adds little beyond length (quintile AUROC 0.687) [to run: within-policy,
fixed-time truncation]. On our own 88 rollouts, where length is uninformative (0.52), a deterministic top-camera check
at rest reaches 94.3 % with no false positives, a robot-trained 27B judge catches 8/8 failures at 2/42 false alarms,
and its base model does as well given the same input; Gemini sliding windows catch 0/8, a world-model surprise 2/8,
an action-conditioned risk head 3/8. Report a length baseline.

Sections: 1 Intro · 2 Datasets and labels · 3 The length baseline (ArmnetBench, own) · 4 Judges at rest · 5 Mid-episode
monitors · 6 What the input format does · 7 Limitations.

Figures: (1) AUROC per detector, both datasets — rows above (today; within-policy `[to run]`); (2) length
distributions per outcome — ArmnetBench metadata (`[to run]`, $0); (3) judge input formats vs result — `g_flora_abl`
README (today); (4) k/8-at-j/42 for every monitor with Wilson CIs (`[to run]`, $0).
