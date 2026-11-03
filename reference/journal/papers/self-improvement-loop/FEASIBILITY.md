# Feasibility — "Closing the loop on a $200 arm": self-improvement from a deterministic verifier, no human in the loop

WS-072, 2026-10-31. Memo only: nothing was run. Sources: `journal/EXPERIMENTS.md`, `projects/so101-multitask-smolvla/README.md`,
the judge reports (`projects/vla-jepa-world-model/judge/results/g9c3/`), `AGENTS.md` §1, the papers mirror. The
"7-day plan of 2026-10-31 morning" named in the prompt is not in the repo (searched `journal/`, `strategy/`): it is
treated as not yet written.

## 1. Claim in one sentence

A SmolVLA policy on an SO-101 improves over K ≥ 4 iterations with no teleop and no human label after iteration 0 —
rollouts judged at rest by a deterministic top-camera verifier, only verified successes added, a ~$3 fine-tune per
iteration, next layouts chosen from the failures — measured as held-out success rising from iteration 0 by at least
a pre-registered margin (e.g. ≥ +15 points on a sealed 20-layout set, CI excluding 0). **No iteration has run: the
claim has no evidence yet, only its components.**

## 2. Evidence in hand (components only)

| component | date | n | result | status | anchor |
|---|---|---|---|---|---|
| verifier G9c-3 (deterministic, at rest) | 10-16 | 88 rollouts | 94.3 % accuracy, **0 false positives** (FP 0/25 [0, 13.3 %] Wilson), FN 5/62 [3.5, 17.5 %], ~1.2 s / episode | narrow fail / operational pass | `judge/results/g9c3/report.md`; Verifier cards row |
| verifier, cold rerun from the public repo | 10-31 | 69 rollouts | same 5 FN episodes, 0 FP | reproduces | WS-067 |
| verifier with an automatic end-rule (G9c-2) | 10-16 | 88 | 29/88 invalid (failing policies rarely return to rest) | fail | EXPERIMENTS |
| fine-tune cost | 10-15 | 653 episodes, 30k steps | 4 h 05 min on an RTX PRO 5000, $2.56 projected (credit −$3.54 incl. stops) | done | `projects/so101-multitask-smolvla` §3 |
| iteration-0 policy (21-task SmolVLA) | 10-16 | 21 tasks on the arm | 10/21; named cube among two **2/6** | measured | same, §2 |
| bowls SmolVLA baseline | 09-29 | 10 | 9/10 | measured | EXPERIMENTS row 23 |
| program-edit loop (G17) | — | — | registered, not run | pending | EXPERIMENTS §3 |
| never imitate failed actions (action loss masked; observations may supervise) — amended 2026-11-01 from "never train on failed rollouts" | — | — | standing rule, enforced by the verifier gate and a λ = 0 check | rule | `AGENTS.md` §1 |
| curation scores (QoQ / CUPID) on the 653 training episodes | 10-24/25 | 69 + 653 | QoQ keeps 26/29 successes in its top half [0.736, 0.964] | measured (placeholder / Gemini validation sets) | EXPERIMENTS §24 rows |

Pre-registered: G9c-3 (bar in the row before the run). Blind: no (operator labels for the 88). Statistic: Wilson
intervals on the verifier counts; nothing on the loop exists.

## 3. Gaps, costed

| gap | resource | queued? |
|---|---|---|
| **automatic reset**, two options to cost against each other before building either (amended 2026-11-01 after HALTER): (a) dump-and-restack fixture; (b) HALTER-style taught reset skills — 2 skills (bowl back to its mark, blue bowl out of pink), 30 demos each, SmolVLA recipe, chosen and verified by the G9c masks' relation | (a) design + build ~3–5 days, parts < €50; (b) 1 teleop session (60 demos) + ~$3 + 1 arm day of reset trials. HALTER's own learned resets reached 48–88 % per task (Table I): (b) is not reliable by default | new |
| **unattended-rollout watchdog** (torque / workspace clamp, stall timeout; 2610.40075's ray search as the clamp design) | ~2 Orin days + 1 arm day of tests | new |
| verifier end-rule for unattended runs (G9c-2's invalid rate is the obstacle) | ~1 Orin day + 50 rollouts | new |
| sealed held-out layout set (20 layouts, hash committed) | 1 h | new (the §29 `LAYOUTS.csv` pattern exists) |
| K = 4 iterations × (40 rollouts + 1 fine-tune + 20 held-out trials) | ~4 × 1.5 arm days = **6 arm days**; 4 × ~$3 = **$12** vast | new |
| LoopWAM-style ablation arm per iteration (same rollouts, quality suffix, failed actions masked; within the amended AGENTS.md §1) — one extra fine-tune + 20 held-out trials | + ~$3 and + 0.5 arm day per iteration (pilot: once) | new (2026-11-01) |
| command–state weighting (CSDW, 2609.33145) as a possible second ablation: leader–follower discrepancy on the 182 demos first | $0 to measure; ~$3 per weighted fine-tune | candidate, not in the pilot |
| a second seed of the whole loop (power) | another 6 arm days, $12 | new |
| failure-driven layout chooser (LLM harness) | ~1 day + API < $1 | harness v1.5 has the slot |
| **total** | **≈ 14–18 arm days, ~4 Orin days, ~$25–30** (+ the ablation arm: + 2 arm days, + $12 over K = 4) | |

At ~2 arm days per week that is 7–9 weeks of arm time.

## 4. Related work

Rewritten 2026-11-01 from `RELATED_CARDS.md` (ENPIRE, LoopWAM, HALTER read in full and their numbers checked) and the
WS-081 cards (papers in the mirror). Entries marked *abstract* are not read in full: their PDFs are not on Drive yet.

| paper | what it showed | the structural difference here |
|---|---|---|
| 2606.19980 ENPIRE | coding agents improve real policies to a 99 % success rate once the environment (safety limits, binary verifier < 150 ms, automated reset) is frozen as immutable APIs; 1 → 8 robots ~2–3× faster at super-linear token cost | 8 YAM stations with an RTX 5090 each and an agent as the improver; here one $200 arm, a fixed fine-tune recipe, the verifier on the Orin beside the policy, and cost per iteration measured |
| 2610.18873 LoopWAM | training on all rollouts with a quality token and failed actions masked (λ = 0, video kept): 23.67 → 43.33 % over 3 rounds, 30 real tasks, 4 embodiments | the failures teach through the video stream; SmolVLA has none, so here λ = 0 removes failures entirely and only the quality token is tested |
| 2609.19413 HALTER | reset by composing a library of taught reset skills over a relation graph: reset 76 % vs 52 % AutoEval, interventions 100 → 25, held-out 74.7 % | RGB-D, a 397B LLM and a Franka; here masks from one RGB camera and 2 skills, or a fixture |
| 2610.05882 Mulligan | starting collection at observed failure states + a value function: +10–34 points over uniform HG-DAgger | human-in-the-loop collection and a value function trained on failures; here layouts from verified failures, no human labels, no failure training |
| 2606.10305 SARM2 | a learned stage-aware reward + on-policy RL for self-improving VLAs | learned reward and RL on failures; here a deterministic verifier with 0 FP and success-only imitation |
| 2610.18792 SEAR | a benchmark of agentic self-improvement with five controls for looking better than it is (Table 2) | many tasks, API agents; here one loop on one arm — SEAR's five controls are this pilot's threats list |
| 2610.09228 Robo-COP | VLA and VLM orchestrator co-evolve during deployment, fine-tuning when failures occur | an orchestrator over skills; here one policy and a fixed recipe |
| 2607.14047 Zero2Skill (*abstract*) | autonomous collect → train → deploy with self-reset and VLM verification | large hardware; a VLM verifier, not a deterministic one |
| 2609.32069 Find Something You Can't Do (*abstract*) | an agent searches for a frozen VLA's failures and trains on them with RL | RL on failures (out by rule here); the search is the layout chooser's idea |
| 2610.40075 safe flow matching from binary feasibility oracles | 100 % constraint satisfaction via an oracle-guided safeguard | the watchdog design reference, not a loop |

Not read here (not in the mirror): Levine et al. 2016–18 arm farm, DAgger (Ross et al. 2011), RECAP / π*0.6, AutoEval
(CoRL 2025), and the other 10 abstract-level entries of `RELATED.md`.

## 5. Objections

**First: training only on your own successes narrows the policy (mode collapse, forgetting).** Answer: the bar is a
sealed held-out layout set measured every iteration, plus the iteration-0 tasks re-tested at the end; the training
set keeps the original teleop episodes (successes are added, nothing is replaced). If held-out success falls, that is
the result.

**Second: "no human" is not true while a human resets the scene, and 0 FP on 88 episodes is not 0 forever.** Answer:
the reset is the first gap in §3 and the paper must not claim "no human" without it; the verifier's FP rate is
bounded (upper Wilson 13.3 %), and a verifier error adds one failed rollout to the training set — the paper reports
the verifier's precision on each iteration's additions by an owner spot-check of 10 %. Also: K = 4–5 iterations and
one seed is little power (§3 costs a second seed); what is new versus the 2016 arm farm is cost and the absence of a
learned reward, not the idea.

**Third (added 2026-11-01): ENPIRE, LoopWAM and Zero2Skill already close the loop.** Answer: yes, and the paper must
not claim "a closed loop". The novelty sentence is cost: a loop whose verifier, reset and fine-tune together cost
under $5 per iteration on a $200 arm and an edge GPU, a number none of them reports (ENPIRE reports tokens and
utilisation on 8 RTX 5090 stations; HALTER reports operator time, not money; LoopWAM reports neither), with the
LoopWAM weighting as the one-factor
ablation.

## 6. Venue

Primary: CoRL 2027 main track (deadline expected mid-2027, to confirm; 8 pages + references, double-blind). Fallback:
a CoRL / RSS 2027 workshop on robot learning at scale or self-improvement. arXiv first: after iteration 2 shows a
direction. Endorsement (cs.RO): no contacts list in the repo; none named.

## 7. Minimal vs full paper

4 pages: one loop, K = 3, manual reset declared as such ("no human *label*" rather than "no human"), the success
curve on the sealed set, the verifier's precision per iteration. Needs: the layout set, ~4.5 arm days, $9. 8 pages:
automatic reset, watchdog, K = 4–5, two seeds, the layout chooser vs random layouts as a control, the program-edit
loop (G17) as a second loop.

## 8. Risk of a null result

Real: the iteration-0 policy is weak on the named-object tasks (2/6) and the verifier only covers the bowls task. If
held-out success does not rise, the paper becomes "what a closed loop on hobby hardware learns when it does not
improve" (narrowing, verifier blind spots, layout chooser effects) — publishable as a workshop paper with the
pre-registered bar, weaker for a main track.

## 9. Recommendation

**go-after-X**, X = a one-iteration pilot with manual reset (iteration 0 → 1 on the bowls task: 40 rollouts, one $3
fine-tune, 20 sealed held-out trials; ~2 arm days). If iteration 1 is not below iteration 0, build the reset and
watchdog and run the full loop. Total for the full paper ≈ 14–18 arm days, ~$25–30. Earliest: workshop version
February 2027; main-track version for CoRL 2027.

## 10. Skeleton

Titles: (1) *Closing the loop on a $200 arm: self-improvement from a deterministic verifier*; (2) *No labels after
day one: verified-success fine-tuning of a VLA on an SO-101*; (3) *A robot that trains itself, for $3 an iteration*.

Abstract (≤ 150 words): Self-improving robot policies are shown with fleets, human labels or learned rewards. We run
the loop on one SO-101 arm and a Jetson: a SmolVLA policy rolls out, a deterministic top-camera verifier with no
false positives on 88 labelled rollouts keeps the successes, the policy is fine-tuned for about $3, and the next
layouts are chosen from the failures. Over [to run] K iterations, success on a sealed held-out layout set goes from
[to run] to [to run]; the verifier's precision on the added rollouts is [to run]. We report what the loop learned when
it did not improve, and what a human still had to do.

Sections: 1 Intro · 2 The loop (verifier, fine-tune, layout chooser, rules) · 3 The verifier (G9c-3, its errors) ·
4 Iterations `[to run]` · 5 Narrowing and forgetting checks `[to run]` · 6 What still needs a human · 7 Limitations.

Figures: (1) the loop diagram (new, no data); (2) held-out success per iteration `[to run]`; (3) verifier confusion
matrix — `judge/results/g9c3` (today); (4) cost per iteration — vast logs (`[to run]`; $2.56 per 30k-step run today).
