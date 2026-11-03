# Chapter 2. Translate the claim

## 2.1 The principle

A paper says: *method M improves metric K on benchmark B, trained at scale S, on embodiment E.* None of B, S or E is yours. The claim you can test is:

> On **my task**, with **my data and compute**, M beats **the cheapest thing that could also explain the result**, by at least **δ**, on **a metric that serves my decision**.

Every bold phrase is a translation, and every translation is a place where honesty is required and usually absent. The paper's claim and your statement are different claims; a fail on yours is not a refutation of theirs unless the translation preserved what mattered.

## 2.2 Translation 1: the task

Benchmarks have properties. Some are obvious (object count, scene clutter), some are not. Three that bit this repository:

**Instruction redundancy.** LIBERO-Goal has multiple valid behaviours for the same scene, so a policy *must* read the instruction to succeed; the TurboVLA paper's Table 3 shows removing language drops LIBERO-Goal from 97.4 % to 11.6 %. On the two-task bowls set, the two instructions share ~94 % of their frames and the arm's own pose leaks the task after the first side-turn. A policy can score 9/10 on bowls while reading the instruction barely at all (SmolVLA's task-swap sensitivity was 7–13°, which is enough; TurboVLA's 0.2° was not, and the arm coin-flipped). A method whose value is "better language grounding" cannot show that value on bowls, whatever the method. Before testing such a method, check that your task *requires* the thing it improves.

**Ceiling.** ACT, SmolVLA and VLA-JEPA all scored 9–10/10 on bowls. A task at ceiling cannot rank methods; it can only disqualify them. The register's note "decide on the hardest primitive" is the consequence: VLA-JEPA vs SmolVLA waits for flips and inside grasps.

**Support.** GR-BC's graph judge (prep queue) is defined over the demonstration support; a rollout in a layout the demos never visited is unreachable, and the judge is silent. A test that only uses layouts from the demos will pass trivially. The held-out-layout split in its protocol exists because of this.

The habit: for each paper, write one sentence on what property of *their* benchmark the method exploits, and one on whether *your* task has it.

## 2.3 Translation 2: the scale

State the ratio as a number, every time. If you cannot, you have not read the implementation section (usually an appendix table).

| Paper | Their scale | Yours | Ratio |
|---|---|---|---|
| TurboVLA LIBERO | 80k steps × batch 256, four RTX 4090 | 35k × 8 (v2) / 35k × 64 (path A), one GPU | 1/128 / 1/4 effective batch |
| RoboJEPA | Meta compute, V-JEPA 2.1-G features | 182 episodes, 192-d W1 | not comparable; different claim |
| Robometer | > 1M trajectories, many embodiments | zero-shot on 88 SO-101 rollouts | n/a: the test is transfer, which is the paper's claim |
| Flora-Rewarder | 27B model, robot-trained | same model, your 50 episodes | 1:1 for inference; the test is transfer |
| D-JEPA | executed outcomes of every candidate from the same start | proxy labels (true chunk vs decoys) | different supervision; your test does not test the paper |

Three consequences:

1. **A fail at 1/100 of the scale is *not proven* unless the claim is about small data.** TurboVLA's claim includes a 260-demo real-world fine-tune from a LIBERO-pretrained checkpoint, which is near your scale, so the faithful reproduction (same init, batch 64) is a fair test of that claim. The from-scratch port at batch 8 was not.
2. **A pass at your scale is stronger than the paper's**, if the claim was "works at scale". Say so when it happens.
3. **When the supervision differs, you are testing a different method.** The D-JEPA entry in NEXT_ANGLES originally described a loss term; the paper trains a relational operator on executed outcomes. The experiment run was "a D-JEPA-inspired operator with proxy labels on frozen W1", and its protocol said so. Its null result says nothing about D-JEPA. Had the protocol not said so, the register would now contain a false refutation.

## 2.4 Translation 3: the metric

Choose the metric that serves the decision, not the one the paper reports. The worked case:

**G1b and the top-1 trap.** The decision was "can the world model pick action chunks". The natural metric is top-1 accuracy among K candidates, and the bar was 0.8. W1 scored 0.60, W2 0.55–0.61. Then the diagnosis: for K = 8, top-1 requires the chosen candidate to beat seven others, so if each pairwise comparison is right with probability p, top-1 ≈ p^7. W2's pairwise accuracy was 0.92–0.93; 0.92^7 ≈ 0.56, which is what it scored. A 0.8 top-1 bar needs p ≈ 0.97. No ranker on 182 episodes reaches that, and more to the point, the decision does not need it: a near-best chunk is fine for control. The metric that serves the decision is **regret**: the outcome gap between the chosen chunk and the best one. Regret ≈ 0 means "always near-best", which is what a planner needs. Regret became the metric of record on 2026-10-13, and the later D-JEPA-operator test was judged on it (0.263, unchanged; ceiling 0.133 even for a perfect top-2 swap).

The general rule: when a metric compounds (top-k of many, precision at a tiny false-positive rate, success over long horizons), compute what per-step performance it implies, and ask whether the decision needs that.

**Other metric choices in the repo and why:**

- Judges: *false positives*, not accuracy. A judge in a training loop must never call a failure a success (that trains the policy on a failure). G9c-3's protocol said accuracy ≥ 95 % and it scored 94.3 %; it was adopted because its false-positive count was 0. The protocol should have said FP = 0 from the start (chapter 4 discusses the adoption-with-miss rule).
- Monitors: *recall at a fixed false-intervention budget* (≥ 5/8 failures at ≤ 1/20 interventions on successes). AUROC is reported because it is comparable across papers, but AUROC does not tell you whether the monitor stops the arm on good runs, which is the cost that matters.
- Chunk boundaries: *median boundary jump* in joint degrees, because jerk is what the operator sees and what strips servo gears; success rate on a ceiling task cannot show it.
- World-model training: *open-loop error at K = 30 relative to copy-last*, plus corruption ratios, because a world model that beats copy-last only at K = 1 has learned to copy.

## 2.5 Translation 4: the embodiment

Not always listed, always present. Three facts about the SO-101 that change what a method can show:

- **Position-controlled, no force sensing.** The commanded joints determine the arm state (chapter 3). Contact is invisible except through the gripper aperture reading, which floors at ~1.1° on a rim (G15-v3). Any method whose value is "better contact reasoning" needs a signal the arm does not have (the CoPRE paper on current-based contact detection is the only route, and the STS3215 bus may not expose current at useful rates).
- **Two fixed cameras, oblique top view.** The deterministic judge works because the top view sees nesting; a method that needs a zenithal view or a third camera cannot be tested.
- **One arm.** Bimanual methods, humanoid methods, and anything about coordination are out of scope regardless of merit.

The embodiment translation is usually what puts a paper in the "not testable here" bin. That bin is not a dismissal: a paper whose claim you cannot test may still change how you read your own results (chapter 17 on bin B).

## 2.6 Writing the translated claim

The template that the protocol's "Question" and "Decision rule" sections answer:

> *Claim as published:* M improves K on B at S on E.
> *Property of B the claim relies on:* ….
> *Does my task have it:* yes / no / partly, because ….
> *Scale ratio:* ….
> *Metric that serves my decision:* …, because the decision is ….
> *Translated claim:* on my data, M beats [baseline] by δ on [metric], with [sample], at [cost].
> *What a fail would mean for the published claim:* refutation / not proven because [scale, task, supervision] / nothing.

The last line is the one people skip, and the one that keeps the register honest. Fill it before the run.

## 2.6a The claim as a row: subject, relation, object, metric, conditions

The template in 2.6 is prose. Since 2026-10-30 the register also stores every tested claim as a **row with a closed vocabulary**, written by hand, one per claim (WS-061, `data/register.yaml` in the site repository):

> **subject** · **relation** · **object** · **metric** · **conditions**
> *latent world-model ranking* · **above** · *action-only baseline* · *G1b top-1* · {model, data, rig, n}

Each paper carries, above its claims, its **goal** (what the authors set out to show, one paragraph, with the sections it was written from) and its **hypotheses**, each linked to the claim row that tests it or marked *not tested here*. The rows are written by reading the paper and the experiment, not extracted by a model; a language-model extraction pipeline was considered and refused, because the value of the row is that someone who read the paper chose its words.

Why the closed vocabulary matters: once the paper's claim and the rig's result are written with the same subject, relation and object, a **contradiction is computable**. The build pairs every paper row with any row on this rig that has the same subject and object and the opposite relation, and lists the conditions that differ. At 2026-10-30: 21 papers, 50 hypotheses, 42 claims (41 tested), **5 contradictions**, 1 agreement:

| Paper says | This rig says | Conditions that differ |
|---|---|---|
| LeWorldModel: latent ranking above action-only (their Push-T planning 90 %) | action-only above latent ranking, 0.68–0.90 vs 0.60 | none |
| DINO-WM: same claim (Push-T 0.90) | same row | model |
| ContactGuard: imagined-latent probe above current-latent probe (0.98 vs 0.84) | current above imagined, 0.82 vs 0.70 | none |
| Robometer: learned reward model above episode length on failure AUROC (VOC 0.92) | episode length above reward model, 0.938 vs 0.818 | none |
| PACT-VLA: cycle equal to naive async on control-step delay (no added delay) | 37 ticks vs 8 | none |

Two rules make the table honest. A row written by this rig and not by any paper is marked "not a paper claim: ours", and it counts in contradictions but never in a paper's verdict. And every number in the "ours" column is re-read from its source file at build time; the build fails on any difference with the text.

"Conditions that differ: none" is the strong case: the claim, as worded by its authors, carries no qualifier that the rig violates. When conditions do differ (DINO-WM: the model), the row is a lead, not a refutation — chapter 2.3's scale rule and 2.5's embodiment rule decide what it means.

## 2.7 What the interviewer will ask

*"You said method X didn't work. The paper reports it does. Who's right?"* The translation table. Name the ratio, the task property, the supervision. Then the honest line: "on my data at 1/4 of their batch with their initialisation, it did / did not; at 1/128 from scratch it did not, and that result says nothing about the paper."

*"Why regret and not top-1?"* The p^7 derivation, in one breath. Then: "the decision needed near-best, not best."

*"What would you need to actually test the paper's claim?"* Have the answer for one or two: D-JEPA needs executed outcomes of every candidate from the same start (a robot day with resets); RoboJEPA needs an off-board CUDA host and a task where object state matters (G7).

## 2.8 Exercises

1. Fill the 2.6 template for each of the eleven prep-queue papers. Where "what a fail would mean" is "nothing", change the test or drop the paper.
2. Compute p^(K−1) for K = 4, 8, 16 at p = 0.90, 0.95, 0.97. Write the sentence you would say to an interviewer who asks why your ranker "only" got 0.6.
3. List every property of the bowls task that makes it easy (ceiling, two objects, colour-separable, static, one instruction redundant with pose). For each, name the method whose value it hides.
