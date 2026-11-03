# METHOD.md: everything behind the work in this repo

<!-- index:start (scripts/ws.py index; do not edit) -->
## Index

Read a section with `sed -n 'a,bp' FILE`; a table row by its line. Generated, refreshed by `ws index`.

- I.1 Start from a decision, not a paper — lines 64–73
- I.2 Translate the claim to your hardware, and say the gap out loud — lines 74–87
- I.3 The baseline is the whole method — lines 88–102
- I.4 Pre-register the bar, then commit it — lines 103–117
- I.5 Isolate one factor; "not proven" when you cannot — lines 118–125
- I.6 Check the measurement before the result — lines 126–141
- I.7 Uncertainty by episode, not by frame — lines 142–155
- I.8 Read the failure, not just the number — lines 156–170
- I.9 Record so nobody re-learns it — lines 171–185
- I.10 The method applied to screening papers — lines 186–203
- II.1 Behaviour cloning, action chunking, and why chunks help — lines 204–211
- II.2 Vision-language-action models — lines 212–221
- II.3 JEPA world models — lines 222–237
- II.4 Judges, reward models, verifiers — lines 238–255
- II.5 Runtime failure monitoring — lines 256–263
- II.6 Data — lines 264–276
- II.7 The harness (v1.5) — lines 277–294
- III.1 Baselines on the arm (09-29 to 10-08) — lines 295–300
- III.2 The first world-model idea and its no-go (10-08) — lines 301–304
- III.3 Training the world model (10-09 to 10-11) — lines 305–308
- III.4 Planning fails against the joints (10-12 to 10-13) — lines 309–314
- III.5 Progress is not distance (10-14, 10-18) — lines 315–318
- III.6 Judging (10-15 to 10-16) — lines 319–322
- III.7 The world model's real role (10-17 to 10-19) — lines 323–326
- III.8 Judges, part two (10-20) — lines 327–330
- III.9 The harness (10-20 to 10-22) — lines 331–334
- III.10 The week of corrections (10-22 to 10-23) — lines 335–364
- Week 1: the method on past gates — lines 365–371
- Week 2: the knowledge — lines 372–378
- Week 3: writing bars — lines 379–385
- Week 4: the site — lines 386–433
- Foundations (read first, D4) — lines 434–443
- Your own results' sources (D3) — lines 444–459
- The harness papers (D2 each; D3 for 21) — lines 460–464
- The prep queue (D3 each, before its bar is final; read the week its run is scheduled) — lines 465–482
- For the Uma postings specifically (D2, read in the two weeks before an interview) — lines 483–490
- Not papers, but read at D3 — lines 491–497
<!-- index:end -->

Written 2026-10-23 for the owner, who asked for one document that explains, from first principles, what was done in this repository, why it was done that way, and what he needs to know to defend every line of it in a research-engineer interview. It is written to be studied, not skimmed: each part ends with exercises, and the exercises are the point. Everything here can be checked against `journal/EXPERIMENTS.md`, the protocols under `projects/`, and the papers in the knowledge base.

Structure:

- Part I: the method (nine steps, with the statistics and the epistemics behind each)
- Part II: the knowledge (what a world model, a VLA, a judge, a chunk, a gate actually are, at the level needed to reason about them)
- Part III: how this repo applied it (every gate, what it asked, what it showed, what it taught, with the mistakes)
- Part IV: how the automation was used, and where it went wrong
- Part V: the study plan and the exercises
- Appendix: the paper list with one line on what each changed

---

# Part I: the method

The method is a way of turning a paper's claim into a decision about what to build next, on hardware that is not the paper's, with a budget that is not the paper's, without fooling yourself. It has nine steps. They are listed in the order you apply them; the reasoning behind each is what you need to own.

## I.1 Start from a decision, not a paper

Write, before anything else: *if this passes I will do X; if it fails I will do Y.* If X and Y are the same, do not run the experiment.

Why. An experiment costs money, teleop time and attention. Its only product is a change in what you do next. If no outcome changes your next action, the experiment is reading dressed as research. This rule cancelled the RoboJEPA 1B re-run on 2026-10-15 (a pass would still have been off-robot only; a fail changed nothing) and it is why G7b and G8 were never run (G7a's fail was sufficient to drop world-model selection).

The subtle case: experiments that are *measurements*, not gates. G2 (how long does one decision take on the Mac), G11-A (latency of a surprise check), the chunk-stability statistic in the prep queue. They have no pass/fail; they produce a number that other decisions need. Label them as measurements in the register. Do not retrofit a bar to a measurement.

Misconception to avoid: "a negative result is still a result, so run everything". A negative result is a result only if it was a pre-registered test of something that could have passed. A negative result from an experiment nobody would have acted on either way is noise with a date.

## I.2 Translate the claim to your hardware, and say the gap out loud

A paper claims: method M improves metric K on benchmark B at scale S. Your statement is: on my task, with my data and compute, M beats the cheapest thing that could also explain the result, by at least δ, on a metric that serves my decision.

Three translations, each a place to be honest:

**Task.** Their benchmark has properties yours may not. LIBERO-Goal has multiple valid behaviours per scene, so language matters there; bowls with two instructions has the decision in ~6 % of frames. A method that needs instruction-dependence to show its value will look useless on bowls for reasons that have nothing to do with the method (TurboVLA, fact 7).

**Scale.** State the ratio. TurboVLA: 80k steps × batch 256 on four 4090s vs your 35k × 8 on a T4: 1/128 of the effective batch. RoboJEPA: Meta compute vs 182 episodes. If the claim is "at scale S this works", a fail at S/100 is *not proven*, not *fail*. If the claim is "this works on small data", a fail at your scale is a fail.

**Metric.** Choose the metric that serves the decision, not the one the paper reports. G1b used top-1 among 8 candidates because that is what planning needs; the diagnosis showed top-1 ≈ pairwise^7 (0.92^7 ≈ 0.56), so a 0.8 top-1 bar needs ~97 % pairwise accuracy, which no method reaches on 182 episodes. Regret (how much worse than the best candidate is the chosen one, in outcome units) is what the decision actually cares about: a near-best chunk is fine. Regret became the metric of record on 2026-10-13.

Exercise: take any paper in the KB and write the three translations in three sentences. If you cannot write the scale ratio as a number, you have not read the implementation section.

## I.3 The baseline is the whole method

Compute the cheapest competitor *before* the model. Every world-model "win" in this repo's history that was later retracted was beaten by one of:

- **Commanded joints.** On a position-controlled arm, the action already says where the arm will be. The joints beat W1 and W2 at ranking chunks (top-1 0.68–0.90, regret ≈ 0, G1b diagnosis). Any claim about predicting *arm* state from images is suspect on this hardware; only object state and contact are informative.
- **Episode length.** Failures take longer or are cut short. Length beats Robometer zero-shot (0.938 vs 0.818 on ArmnetBench; 0.86–0.97 vs 0.69 median on held-out tasks in the G9b re-analysis). Any judge that does not beat length has learned length.
- **Policy identity.** When one policy produced most failures (TurboVLA in C1 and in the REAP secondary set), "which policy ran" predicts failure. Any monitor evaluated on a mixed-policy set must be reported on the single-policy subset too.
- **Copy-last.** For a world model: predict that the next latent equals the current one. G0's bar was "beat copy-last at K = 30" (0.31 vs 0.89).
- **The mean of N samples.** For best-of-N: averaging the candidates. H0 showed the mean captures 45 % of the oracle gain with no model at all.
- **The current component.** For a replacement: the thing already in the stack (G9c-3 for judges, prefix overwrite for chunking).

The question to ask of every paper: *what trivial thing did they not compare against?* Most do not report episode length for a judge, or commanded joints for a latent planner, or the sample mean for a reranker. That absence is the first thing your test adds.

Why this works. A cheap baseline is a null hypothesis you can compute in an afternoon. If the method does not beat it, you learn the method is not doing what its paper says, or your data lets a shortcut through. Either way you stop before spending the robot day.

## I.4 Pre-register the bar, then commit it

Before the first number exists, write into a protocol file: the metric, the split, the sample, the pass number, the fail number, and the decision each outcome triggers. Commit it. Everything decided after that commit is labelled *exploratory*.

Why. Without a committed bar, every result can be rescued: move the threshold, change the split, drop the outliers, call 86.4 % "close to 95". The garden of forking paths: with enough analytic choices, something always passes. The commit hash removes the choices. It is also the only thing that makes a *fail* publishable: a reviewer can see the bar was set before the number.

Rules this repo uses:

- A protocol is "frozen" when committed with its bar. The commit hash appears in the register row.
- A result that narrowly misses may still be adopted, but only with the miss written down (G9c-3: 94.3 % vs 95 %, adopted because its false-positive count was 0, which was the property the decision needed; the protocol said accuracy; the lesson is that the bar should have been false positives from the start).
- Never rewrite a row. Add a correction row (G11-A, G12, G9a-AB, G9b). The register is a log, not a scoreboard.
- Amendments to a protocol are themselves committed before the number they affect (D-JEPA protocol amendment 2, G7 cube amendment).

Exercise: pick G6a. Read its protocol. Write down what you could have changed after seeing −0.32 to −0.50 to make it pass, and why the commit prevents each.

## I.5 Isolate one factor; "not proven" when you cannot

An ablation changes one thing. Flora vs its base model Qwen3.5-27B with the *same* input showed the robot fine-tune was not what mattered; had the input changed too, nothing could be concluded. H0 compared reranking against averaging the *same* eight samples. G13b compared prefix overwrite against exact RTC on the *same* 254 boundaries.

When the confound cannot be removed, the honest verdict is *not proven*. C1 (2026-10-11): policy identity alone scored 0.76 because TurboVLA made most failures; the latent probes could not be assessed. REAP (2026-10-23): the risk head's 23/25 on the 69-rollout set is TurboVLA failing visibly early; on the 8 non-TurboVLA failures it catches 6 but at 7/42 false interventions; and the onsets came from a VLM that was 110 and 289 frames late on two episodes. Verdict: not proven pending clean onsets, not fail.

Three words, three meanings: *pass* (bar met), *fail* (bar missed and no confound explains it), *not proven* (a named confound could explain the miss). Using *fail* for a confounded result is the most common way to throw away a method that works.

## I.6 Check the measurement before the result

Two gates in this repo were voided because the measurement was wrong, not the method:

- **G11-A** (2026-10-18): 12.78 ms per surprise check on the Orin, pass. The next day: the benchmark loaded `weights_final.pt` with `strict=False` and a key remap written for a different transformers version; 192 encoder keys silently failed to load; the ViT was random. The timing stood (same architecture); every latent value was void.
- **G12** (2026-10-18): REKI 2.4B at 318 ms on the Orin, fail. The review: the script timed a random-init Qwen2.5-1.5B with no vision tower and a hand-written random expert. The parts did not sum (135.4 + 230.3 ≠ 318.2 ms). The 318 ms measured nothing.

The checks now run before every Jetson gate (`scripts/check_strict_load.py`, GATES_RUNBOOK §0):

1. **Strict load.** `load_state_dict(strict=True)` or stop. Missing and unexpected keys printed and counted; zero of each or the run does not start.
2. **Parts must sum.** Time the components and the total; if they disagree by more than measurement noise, something is not running.
3. **Parity with the reference.** Converted or ported models (GGUF, vla.cpp ports) compared to the PyTorch reference on fixed inputs *with shared randomness* (same noise tensor for flow matching). The GGUF SmolVLA check: median max |Δ| 0.80° against a 1° line, one outlier at 11.4° reported, PyTorch's own noise-to-noise spread 12.2° reported so the line has context.
4. **Smoke run whose output you expect to be meaningless.** A 4-step checkpoint through the full gate pipeline. If its numbers look reasonable, your pipeline is not measuring the model.

Why. A measurement bug produces a plausible number with the wrong sign of surprise: passes feel earned, fails feel informative. Nothing downstream can detect it. Only an independent check of the instrument can.

## I.7 Uncertainty by episode, not by frame

Frames within an episode are correlated: the same lighting, the same layout, the same policy sample. Treating 41 000 frames as 41 000 independent samples gives confidence intervals that are tens of times too narrow.

Rules:

- **Bootstrap over episodes**, 2 000 resamples: resample episodes with replacement, recompute the metric on all their frames, take the 2.5 and 97.5 percentiles.
- **Paired comparisons use the same resamples.** To compare scorer A and scorer B on the same episodes, compute A − B within each resample and report the CI of the difference. "A = 0.86 [0.80, 0.91], B = 0.82 [0.75, 0.88]" says nothing about whether A > B; "A − B = 0.04 [−0.02, 0.10]" says the data cannot tell them apart.
- **Small-n success rates.** With n = 10 trials, 9/10 vs 8/10 is noise. Wilson 95 % interval for 9/10 is roughly [0.60, 0.98]; for 8/10 roughly [0.49, 0.94]. To detect a 10-point gap (say 80 % vs 90 %) at conventional power you need on the order of 200 trials per arm; at 20 per arm you can detect only gaps of ~30 points. This is why the prep queue says "n = 6 per cell cannot resolve less than ~30 points" and why the 10-16 gate reports but does not decide.
- **Report n in every table.** A row without n is not a result.
- **Multiple comparisons.** When many gates run on the same 69 rollouts, some will pass by chance. The defence is not a correction formula (the gates ask different questions); it is pre-registration (I.4) and replication on fresh rollouts (W1-A was the fresh-rollout check that undid G10's thresholds).

Exercise: write the episode bootstrap in 30 lines for the C1 probe data and reproduce the policy-identity AUROC of 0.76 with its CI. Then compute the paired CI of (imagined − current). You will see why "not proven" was the only honest word.

## I.8 Read the failure, not just the number

A fail with a mechanism is worth more than a pass without one. The TurboVLA chain is the model:

1. Rollouts fail → check the pipeline (normalisation, camera order, task plumbing): ruled out.
2. The arm does the same thing for both instructions → task-swap probe (change the instruction, measure the action change): 0.1–0.25° vs 7–13° for SmolVLA. The instruction is ignored.
3. Is the text path dead? → attention mass on text 7–21 %, text-token gradients 24–33× image patches, the colour token differs 46–63 % after projection. Alive.
4. Where is it lost? → per fusion layer, the instruction-specific part of the text→vision update is 1–4 %; the action chunk differs ~0.5 %. Washed out in fusion.
5. Can the decoder read it at all? → auxiliary "which bowl" head on the decoder states reaches 98 % in 1k steps. Represented, not used.
6. Why would training not use it? → on paired pre-grasp frames, swapping the instruction changes the L1 loss by 0–4 %; the side decision lives in ~6 % of frames and the arm's pose already leaks it. The cheapest fit is copycat.

Six probes, each one answering "where exactly does the signal die". The final explanation (fact 7) is about the data and the objective, not the architecture, and it generalises: any short-horizon L1 head with state input on teleop data is at risk. That generalisation is the research contribution; the bug was just the way in.

And then the correction: the port itself differed from the paper (random-init fusion, state inside fusion, different interaction module). Fact 7 stands as a data property; "TurboVLA ignores instructions" does not stand until the faithful reproduction (path A, running now) says so. Reading the failure well includes reading your own code as a suspect.

## I.9 Record so nobody re-learns it

One row per gate in `EXPERIMENTS.md`: date, ID, question, bar, result, verdict, cost, link. One line per lesson in `LESSONS.md`, with the evidence named. Never rewrite; add correction rows. Chat-only facts marked *chat* until a file records them.

Why. Two weeks in, the repo already had results that contradicted earlier chat conclusions (G9b "running since 10-16" never ran; the first G9b summary counted 400 teleop demos as successes). The register is what let those be found. A lab notebook that can be edited is a lab notebook that will be edited.

What the register does not do: it does not tell you what to believe. That is Part III.

**Two readers, one page (WS-075, 2026-10-31).** Every register post opens with a plain layer for engineers who do not
read papers: a three-sentence lede ("They said X. On our arm we checked by doing Y. It did / did not hold, by this
much."), the result plot, an "If you build robots" and an "If you do research" box, and first-use definitions from a
closed glossary. The layer adds no claim the post does not carry; "they said" comes only from the register's claims
rows as checked against the PDFs; the post's own text is hash-checked unchanged at every build. Rules:
`journal/STYLE_POSTS.md`; new drafts start from `journal/drafts/_TEMPLATE_post.md` (sections 0a–0c).

## I.10 The method applied to screening papers

The same steps, compressed, decide whether a paper is worth a test:

(a) Which decision could it change here? (I.1)
(b) Does the claim survive translation to one arm, two cameras, ≤ $15? State the scale gap. (I.2)
(c) What is the cheap baseline, and did they run it? (I.3)
(d) Is there code, so parity is possible? (I.6)
(e) Can the bar be written as a number today? (I.4)

A paper that fails (a) or (e) is reading. A paper that fails (b) is a short "not testable here, and here is what it changes in my reading of my own results" post. The rest are tests.

---

# Part II: the knowledge

What you need to understand, at the level needed to reason about the gates, not to re-derive the papers. Each section ends with what the gates in this repo taught about it.

## II.1 Behaviour cloning, action chunking, and why chunks help

A policy π(a | o) is trained to imitate demonstrations by regression (L1, MSE) or by a generative head (diffusion, flow matching). **Action chunking** predicts a sequence a_{t..t+H} from o_t and executes some prefix before re-querying. Chunks help for three reasons that are usually conflated: temporal consistency (one decision, not H noisy ones), horizon reduction (fewer decisions per episode compounding errors), and covering inference latency (the arm has something to do during the next 238 ms). The paper in the prep queue (2610.01626) argues the benefit depends on an environment-stability assumption over the chunk: static table, long chunks fine; moving bricks, not.

**Chunk boundaries** are where chunked policies jerk: the new chunk is sampled fresh and its first action does not match where the arm actually is. Real-time chunking (RTC) fixes this with inference-time guidance toward the previous chunk's tail; prefix overwrite fixes it by copying the tail directly. On your data prefix overwrite cuts the median boundary jump 72–74 % at +0.4 ms; exact RTC 21–41 % at twice the latency (G13b). Register tokens (REKI) attack a different symptom, oscillation between poses (24 % of failed episodes, G14).

What the gates taught: on a static tabletop, the engine and the boundary handling matter more than the policy architecture (ACT, SmolVLA, VLA-JEPA all at 9–10/10; 4× latency difference between engines).

## II.2 Vision-language-action models

A VLA = a vision-language backbone producing tokens + an action head conditioned on them. SmolVLA (450M): a small VLM, frozen in your fine-tunes, and a flow-matching action expert. VLA-JEPA: Qwen3-VL-2B + a DiT action head, with a JEPA world-model loss at training time only. TurboVLA: no LLM; DINOv3 + BERT + a Grounding-DINO-style bidirectional fusion + an ACT-style decoder.

**Flow matching** for actions: the head learns a velocity field v(x_t, t | o) that transports Gaussian noise x_1 to the action chunk x_0 along x_t = t·noise + (1 − t)·actions; inference integrates it in N steps (10 by default). Fewer steps = faster and less exact; shared noise is what makes two implementations comparable (parity, I.6). The fewer-steps paper in the prep queue (2609.21216) changes the integration schedule, not the model.

**Instruction conditioning** is not free. A policy can achieve low training loss while ignoring the instruction if the data makes the instruction redundant with the state (fact 7). The sensitivity gate (`sensitivity_gate_v3.py`: start reach, task-swap, per-layout) is the five-minute check that catches this before hardware time. It generalises to any policy with state input and an action space close to the state space.

What the gates taught: data properties (idle starts, instruction-redundant frames) explain more policy failures than architectures do. Fix the data first.

## II.3 JEPA world models

A **joint-embedding predictive architecture** learns an encoder E and a predictor P such that P(E(o_t), a_t) ≈ E(o_{t+1}), in latent space, with no pixel reconstruction. Collapse (everything maps to one point) is prevented by a target network updated by EMA, by variance/covariance regularisers (VICReg), or by SIGReg (LeWM). The latent is "action-conditioned": shuffling or shifting the actions should raise the prediction error (G0 checked this: timing shifts raised error 1.99× and 1.58×).

Your two models: **W1** (LeWM recipe, 192-d pooled latent, two views, trained from pixels end-to-end, $3.30); **W2** (frozen DINOv2-S patch tokens, a learned dynamics head; arm-pose R² 0.92 vs W1's 0.57; 1.8 s per decision on the Mac). Spatial features beat a pooled vector for anything about the arm; never PCA-compress a JEPA latent (information sits in low-variance directions).

Three roles a world model can play, and what the gates said:

- **Planner / ranker** (imagine each candidate, pick the best): fails here. G1b top-1 0.60 vs 0.80 bar; commanded joints 0.68–0.90. A ranking operator on frozen W1 (D-JEPA-inspired, proxy labels) leaves regret at 0.263; even a perfect top-2 swap floors at 0.133 with 8 candidates. Mechanism: on a position-controlled arm the actions determine arm state; 182 episodes without counterfactual failures teach nothing about object state the joints do not already imply. D-JEPA's actual method (a relational operator trained on *executed* outcomes of every candidate from the same start) needs data you do not have.
- **Progress / stall signal** (latent distance to goal falls monotonically): fails. Spearman −0.32 to −0.73 against a −0.8 bar; 32–52 % false stalls. Trajectories loop in physical space. A progress signal must be trained on progress labels (TD-JEPA: ~0.65 raw distance vs 0.95 learned cost).
- **Verifier** (read the outcome at rest; flag surprise mid-chunk): the readout passes (G10: AUROC 1.000 at the judge timestamp; anticipation 30 steps early beats policy identity by +0.18) but the live thresholds do not transfer to fresh rollouts (W1-A: the reflex fires on 42/42 successes; refit catches 3/25 failures at 1/44 false alarms; W1-C1: imagined latent 0.70 vs current 0.82). The world model is in shadow mode: scored and logged, never allowed to stop the arm.

The honest summary, which is the thesis of the site: on cheap position-controlled hardware with a few hundred episodes, a JEPA world model earns its place as an at-rest verifier and a logged signal, not as a planner or a live monitor. The one open test is object-state ranking (G7), where joints are blind.

A caution from a paper in the KB (2610.19375): one-step prediction error and planning success correlate at r ≈ −0.01 across LeWM checkpoints, and a state-grounding head is indistinguishable from five more epochs of training. So "lower loss" is not "better world model" and any grounding-head gain needs a matched continued-training control. That amendment is in NEXT_ANGLES §7.

## II.4 Judges, reward models, verifiers

A judge maps a rollout (video, state, actions) to success/failure or a progress score. Four tiers, from v1.5 §1.1, in decreasing credibility and increasing generality:

1. Physical measurement (a scale, a count, a limit switch).
2. Deterministic perception at rest (G9c-3: HSV masks + geometry, 0 false positives, 94.3 % accuracy; all errors are recordings that run past the operator reset).
3. Learned or VLM judge (Robometer, Flora, Qwen, Gemini).
4. Owner label (ground truth, slow).

What the gates taught about tier 3:

- Zero-shot reward models trained on other robots do not transfer: Robometer-4B 0.58–0.62 on your rollouts, 0.818 on ArmnetBench where length alone is 0.938. The failure mode that matters (wrong object) is an identity question a trajectory-comparison model has no reason to answer.
- VLM judges fail when asked mid-episode (sliding window: 0/8 failures caught; checkpoint calls: 41/42 successes flagged, because nested bowls look like "blue alone" and the post-grasp checkpoint precedes the lift) and succeed when asked once, at the end, with the whole episode and both views (Flora 8/8 at 2/42; Qwen3.5-27B 7/8 at 2/42 with the same input; Qwen3.5-9B fails). The input format and the model size did the work, not the robot fine-tune.
- A judge whose false positives are not zero poisons a training loop: every success it mislabels becomes a failure in the next fine-tune. That is why the decision rule prefers a narrow deterministic judge with 0 FP over a general one with 2 %.
- Coverage shift (2610.24801): as a policy improves, the failures that remain are harder to see, so a fixed judge's false-accept rate rises. A judge that passed on this month's failures must be re-checked on next month's.

The rule written into v1.5: tier 3 never decides alone; it pairs with a tier-2 verdict and earns its place per task through a report card (FP, FN, ambiguous rate against owner labels).

## II.5 Runtime failure monitoring

Different problem from judging: predict failure *before* it finishes, with a budget of false interventions. Signals tried here: W1 prediction error (surprise), cosine drift of the latent, a risk head on [latent, commanded actions] (ReAP-style), episode length, elapsed time, conformal bands on the surprise trace (FAIL-Detect / SAFE style).

What the gates taught: on 50 fresh rollouts no monitor clears "≥ 5/8 failures at ≤ 1/20 false interventions". The risk head ranks well (AUROC 0.93) but intervenes on 7/42 successes, and its warnings come ~0.9 s after onset. Two confounds remain (TurboVLA dominates the extra failures; the onsets are VLM-labelled and sometimes late), so the verdict is not proven. The conformal band as registered was not conformal across time because μ and σ were fit on the same successes.

A monitor's latency budget on the Orin: a W1 check is 17.6 ms (19.1 under vla-server load), fine inside a 33 ms tick; a 25-step imagination is 153 ms, so it must run off the control thread; running the monitor slows the policy's chunks from 238 to 304 ms. These numbers are in NEXT_ANGLES §2 and v1.5 amendment D60.

## II.6 Data

The scarce resource is teleop time (~2 robot days a week). Everything else follows:

- **Idle frames teach freezing** (TurboVLA never left rest until idle starts were trimmed). Trim every dataset the same way.
- **Instruction-dependent behaviour needs data where the instruction matters** (fact 7). Two tasks that share 94 % of their frames teach a copycat.
- **Failures train the world model, the cost and the judge, never the policy** (AGENTS.md rule 1.2). Label one key press per episode, right after it.
- **Publish rollouts and checkpoints to the Hub the day they are produced.** W2 was lost because it lived only on a vast instance (LESSONS 24, closed 2026-10-23: W2′ and W1 on the Hub). Rollouts went to the Hub on 10-14 because the Jetson disk filled up; that is the wrong trigger.
- **Camera choice changes the judge** (Robometer top view 0.861 vs front 0.798).
- **Missing data is not random** (22 % of ArmnetBench episodes lost to one OOM were two whole tasks; re-scoring them flipped a qualified pass to a fail).

Curation signals to be audited in the prep queue: trajectory smoothness (likely ranks skill, not success, once length is controlled), state criticality (should recover the 6 % decision band if it works), outcome-based weighting vs filtering vs conditioning.

## II.7 The harness (v1.5)

The current architecture is not a model; it is a contract around models. Station = harness; a coding agent is the operator; scripted motions and a small VLA are interchangeable executors; the verifier is a per-task plugin with a stated tier; every trial is a labelled record.

The three contracts and why each exists:

- **Two-field results** (`executor_status` × `verdict`, `ok` = both): because an executor that writes its own verdict will grade itself; the verifier never sees the agent's text (RegenHarness, SEAR).
- **Idempotent dispatch** (`key = (run, action_id, attempt)`): because an agent that retries after a lost tool response will move the arm twice.
- **Budgets the agent cannot raise**: because an autonomous loop that can escalate its own retries will.

The world model sits in shadow mode (II.3). The data flywheel writes every trial in LeRobot v3 with task string, layout id, executor version, verdict, evidence; `demos/` is a view over tier-1/2 satisfied trials; the next teleop session is planned from failed layouts first (≤ ⅓ of the budget), then farthest-point coverage. A policy replaces a scripted executor only through an adoption protocol (G24). None of v1.5 has run end to end on the arm; it is specified and partly built (284 tests).

---

# Part III: how this repo applied it, gate by gate

The story in order, with what each step asked, what it showed, what it taught, and the mistakes. Dates are the register's. Read it with `journal/EXPERIMENTS.md` open.

## III.1 Baselines on the arm (09-29 to 10-08)

Question: does a learned policy run reliably on this hardware, and can I measure it the same way twice? ACT 10/10 (Mac), 9/10 (Orin, 59 ms); SmolVLA 9/10 (974 ms PyTorch → 390 → 238 ms via vla.cpp); VLA-JEPA 9/10 (355 ms, $9.7 to fine-tune); TurboVLA 0/3, closed. Taught: the task is at ceiling (cannot separate policies); the engine is the lever; operator judgment is fine for 10 trials and will not be for 100.

Mistake: the gate harness recorded the operator's key press but the recordings ran past the reset, which later cost G9c 12 "errors". Fix the recording boundary before the judge, not after.

## III.2 The first world-model idea and its no-go (10-08)

H0: do VLA-JEPA's 8 samples vary enough to pick from? Pre-registered: oracle/mean ≤ 0.7, latency ≤ 1.5×, baseline < 85 %. Result: 0.76, 1.84×, 90 %, and the mean of 8 gets 45 % of the oracle gain. No-go the same day. Taught: check the premise of a method before building it; the cheapest competitor was the sample mean.

## III.3 Training the world model (10-09 to 10-11)

Eleven environment problems, $0.28 of smoke runs, a 5-epoch model with effective rank 5.8 (too short to judge). G0, 100 epochs, $3.30 (+$2.96 failed attempts): error 0.31 vs copy-last 0.89, reacts to action timing, rank 24–30, grasp-onset probe 0.94. Pass. C1 the same day: policy identity 0.76 beats every probe; not proven. Taught: debug on the Mac, rent for the run; compute the confound before the model.

## III.4 Planning fails against the joints (10-12 to 10-13)

G1b: W1 best top-1 0.60; W2 0.55–0.61 despite a far better representation; W3 not run by the decision rule. Diagnosis: top-1 ≈ pairwise^7; action-only baseline 0.68–0.90, regret ≈ 0. Regret adopted. RoboJEPA: 526 ms for two views on the Orin, off the robot; zero-shot borderline on 30 clips. Taught: prediction quality is not decision quality; on this arm the joints are the kinematics model; only object state and contact are open ground.

Mistake avoided: the RoboJEPA 1B re-run was cancelled because neither outcome changed the build.

## III.5 Progress is not distance (10-14, 10-18)

G6a: monotonicity −0.32 to −0.50 vs −0.8; 32 % false stalls; the done detector's AUROC 1.00 was an easy contrast (at-rest negatives). G6b (H-JEPA level 2): −0.73, 52 % false stalls; hierarchy dropped. Taught: a progress signal needs progress labels; easy contrasts prove little.

## III.6 Judging (10-15 to 10-16)

G9a-AB: Robometer zero-shot on ArmnetBench, qualified pass at 0.861 on a partial set, fail at 0.818 once the 462 OOM-lost episodes were re-scored; length 0.938. G9a-own: 0.58–0.62. G9c → G9c-2 → G9c-3: the deterministic judge, 0 false positives, adopted at 94.3 % against a 95 % bar with the miss written down. Taught: missing data is not random; a judge must beat length; the bar should have been false positives, which is what the decision needed.

## III.7 The world model's real role (10-17 to 10-19)

G10 (retrospective, on seen rollouts): readout 1.000, anticipation +0.18, surprise spikes 1.82× / 2.58×. Architecture rewritten around veto + monitor. G11-A 12.78 ms, pass; voided the next day (random encoder). Then the fresh-rollout day (10-19): W1-A, G11-B-cal, W1-C1, W1-C2 all fail; G15 / G15-v2 / W1-B stage trackers fail; judge τ and G16-B VLM judges fail; G13b prefix overwrite passes offline and in vla.cpp. v1.1: verify at rest only, world model in shadow mode. Taught: a retrospective pass on seen data is a hypothesis; the fresh-rollout gate is the test; six mid-episode verifiers failing on one day is a design signal, not six bugs.

## III.8 Judges, part two (10-20)

G-flora: 8/8 at 2/42, AUROC 1.00, deterministic, $0.70. G-flora-abl: Qwen3.5-27B passes with the same input, 9B fails, Gemini Flash narrowly fails. Taught: isolate one factor; the format and the size did the work; the fine-tune is a nice-to-have.

## III.9 The harness (10-20 to 10-22)

v1.2 → v1.5 in three days, driven by the LEGO task and nine harness/self-improvement papers. Specified, partly built, not yet run on the arm. Taught: the agent stays out of the motion loop; a task is a plugin; the verifier is tiered.

## III.10 The week of corrections (10-22 to 10-23)

- G9b never ran (the 10-16 job died at preprocessing); the "running since 10-16" line was wrong for a week. The re-analysis of the baseline found the first summary had counted 400 teleop demos as successes.
- The D-JEPA entry in NEXT_ANGLES misdescribed the paper (it trains a ranking operator on executed outcomes, not a loss term); the corrected experiment on frozen W1 found regret unchanged and a 0.133 ceiling.
- The TurboVLA port differs from the paper in the fusion module (random init, different block, state inside fusion); the released checkpoint with upstream code passes a task-swap control the port fails 20–55× over. Faithful reproduction running.
- W2's checkpoint is lost (lived on a destroyed vast instance); W2′ retraining prepared.
- REAP: no monitor passes; two confounds; not proven.

Taught: the register works (every one of these was found by re-reading it against the files); the automation produces plausible wrong lines at the same speed as right ones; the owner is the only check.

---

# Part IV: how the automation was used, and where it went wrong

This section is for you to be honest with yourself about, because an interviewer will ask "what did you do and what did the tools do".

**What Claude Code did well.** Plumbing: the Jetson Docker image, vast launchers with safety deadlines, the GGUF parity check with shared noise, the strict-load check, the two-environment path A build, the episode-bootstrap scorers, the register bookkeeping. These are hours of careful work that produce exactly what was asked, and that you could have done more slowly. Also: catching its own errors when the method told it to (the G9b teleop-as-success bug, the 17-vs-13 TurboVLA count, the fail-fast import order).

**What it did badly, and why.** (1) It executed descriptions of papers that were wrong (D-JEPA as a loss term) because the description came from me, not from the paper; the method's step 2 (read the implementation section) was skipped upstream. (2) It produced protocols faster than you could read them; the fifteen entries in NEXT_ANGLES were written in one evening, which means nobody had thought about each for more than a few minutes. (3) It reports verdicts in the register's voice, which makes them look settled; "fail" for REAP would have stood if nobody had asked about the onset labels.

**What I (the assistant in the chat) did badly.** Built three layers of automation in one morning (weekly scan → work package → repo writer) after you had said you wanted to slow down. Described D-JEPA from memory. Wrote the first NEXT_ANGLES with a "running since 10-16" line copied from a stale file. The pattern: speed presented as help.

**The correct division of labour**, which this document and the prep queue now follow: the tools build instruments and run pre-registered procedures; you choose the question, read the paper, write the bar, read the failure, and sign the verdict. Every verdict that goes on the site under your name should have passed through your hands at steps 1, 4 and 8 at least.

---

# Part V: study plan and exercises

Four weeks, four evenings a week, one hour each. Each exercise produces a file in `journal/study/`; commit them. The point is that you can redo the work, not that the work gets redone.

## Week 1: the method on past gates

1. **Redo G1b regret by hand.** Load W1 and the frozen G1b anchors; write the regret computation in ≤ 50 lines without looking at `g1_diag.py`; reproduce 0.263. Then compute the action-only baseline. Write 200 words on why the joints win on this arm.
2. **Episode bootstrap from scratch.** On the C1 probe outputs, write the 2 000-resample bootstrap and the paired CI of (imagined − current). Reproduce the "not proven" verdict numerically.
3. **Read G6a's protocol and list the forking paths.** Five things you could have changed after seeing −0.32 that would have produced a pass, and what in the commit prevents each.
4. **Rewrite G9c's bar.** The decision needed zero false positives; the protocol said 95 % accuracy. Write the bar that should have been written, and the sentence that justifies adopting 94.3 % under it.

## Week 2: the knowledge

5. **One-page derivation of why JEPA collapses and what SIGReg / VICReg / EMA each do.** From the LeWM and V-JEPA 2 papers in the KB. Then: which of the three does W1 use, and where in `train_lewm_mv.py`.
6. **Flow matching on paper.** Write the forward path x_t, the velocity target, and the Euler step; explain why shared noise is required for parity; explain what changing the schedule (2609.21216) does and does not change.
7. **The TurboVLA chain, from memory.** Six probes in order, each with its number. Then the four divergences of the port from the paper. Then the one-sentence honest claim that survives.
8. **Wilson intervals and power.** Compute by hand the Wilson 95 % interval for 9/10, 8/10, 18/20, 54/60; then the n per arm needed to detect a 10-point gap at 80 % power. Write the sentence you would put in a post with n = 20.

## Week 3: writing bars

9. **Read P² (2610.31613) §IV–V yourself.** Write the bar on paper before reading the protocol Claude Code wrote. Compare. Where you differ, decide who is right and why; amend the protocol if it is you.
10. **Same for GR-BC (2610.20504)** with attention to the held-out-layout question: is a judge that only works inside the demo support a judge?
11. **Same for the smoothness audit (2610.17427)**, which is the one prep-queue paper closest to the Uma Data JD. Write the partial-correlation design yourself first.
12. **Rewrite one prep-queue bar you disagree with**, commit the amendment with your reasoning, before its run.

## Week 4: the site

13. **Re-read every post on ravediamond.github.io with Part I open.** For each: which method step does it demonstrate, which does it skip, and is the verdict word right (pass / fail / not proven). List the fixes.
14. **Rewrite the TurboVLA post** once path A reports: three failures, one method (the sensitivity gate), the port divergences, the faithful result, one honest limit (scale). 1 500 words, two plots.
15. **Write the method post**: Part I of this document, in your words, 1 200 words, with your gates as the examples. This is the post that tells a hiring manager who did the thinking.
16. **Interview drill.** For each of the eight results on the landing page, a 60-second spoken answer: claim, baseline, bar, number, mechanism, what it changed. Record yourself. The ones you cannot do without notes are the ones to redo from exercise 1.

---

# Appendix: the papers, one line each on what they changed here

- LeWorldModel (2603.19312): the W1 recipe; stable JEPA from pixels; SIGReg.
- LeWM / ContactGuard (2608.13438): the multi-view recipe and the event-anchored grasp check (W1-C1 failed it).
- V-JEPA 2.1 (2603.14482): the dense features RoboJEPA scales on; too large for the Orin.
- RoboJEPA (2610.10515): scaling of latent world models; Gate A/B no-go on device; the scaling angle in NEXT_ANGLES §5.
- D-JEPA (2609.24749): decision alignment ≠ prediction accuracy; the ranking-operator test on frozen W1 (null).
- The Planning Limits of Latent World Models (2609.39235): the rollout must reach the target; explained G1b's below-chance short horizons.
- H-JEPA (2610.06805): tested as G6b, dropped.
- One-step fidelity ≠ planning (2610.19375): the matched continued-training control in §7.
- Physical-state grounding ×2 (2608.06799, 2609.03565): the state-head angle in §7.
- CF-JEPA (2610.00727): the lighting-robustness angle in §8.
- VLA-JEPA (2602.10098): the policy; the future-frame leak fix you merged upstream (#4381).
- SmolVLA (2506.01844): the workhorse policy; 450M; frozen VLM in fine-tunes.
- TurboVLA (2607.27205): the port, the diagnosis, the faithful reproduction.
- Real-Time Chunking (2506.07339): the exact-guidance arm in G13b.
- REKI (2610.22178): register tokens, the oscillation metric (G14), the voided G12.
- FAIL-Detect (2503.08558), SAFE (2506.09937): conformal bands; W1-C2 failed as registered.
- ReAP (2610.16309): action-conditioned risk head; §2, not proven.
- Robometer (RSS 2026): zero-shot fails on SO-101 data; the G9b re-analysis.
- Flora-Rewarder (NVIDIA): passes; non-commercial; the ablation that showed format > fine-tune.
- No Free Checker (2609.09250): the verifier taxonomy behind the v1.5 tiers.
- RegenHarness, SEAR, Robo-COP, Mulligan, ReTrace, Trace-to-Architecture, Embodied Turing Machines (2609.27612, 2610.18792, 2610.09228, 2610.05882, 2610.15814, 2610.16725, 2610.12369): the v1.5 contracts.
- HexaShell (2610.23479): 739 s of agent time for 26 s of robot time; the agent stays out of the loop.
- SEVO / ARRO / PEEK (2605.11114, 2509.18282): the perception-overlay work for LEGO.
- ArmnetBench (2607.24481): the 2 099 labelled rollouts; the scoring protocol to re-read before FailBench.
- FailBench (2609.03611), RAFAIL (2609.18324), Robot-WATCHDOG (2610.13403), smoothness audit (2610.17427), state criticality (2610.17709), chunking stability (2610.01626), GeoAAC (2609.20776), fewer-steps flow (2609.21216), SMILE (2608.29432), RESTA (2610.18163), quality-aware pretraining (2610.18869): the prep queue, 2026-10-23.
- P² (2610.31613), GR-BC (2610.20504), coverage shift (2610.24801), VLA-Rolling (2610.25043), PICES (2610.19656): queued from the first weekly digest.

---

# Appendix B: the reading list, with how deeply to read each

> Superseded on 2026-10-24 by the book's Appendix R (`journal/book/91-reading-list.md`), which adds the papers of chapters 12c, 12d, 13b and 14b and marks KB status. Kept for the record.

Four depths. **D4, derive:** read every section, reproduce the key equation or algorithm on paper, be able to explain the training objective and the main table from memory; 3–4 hours. **D3, implement-level:** read method, experiments and appendix for implementation details; know the hyper-parameters, the ablations and the baselines they ran and did not run; 1.5–2 hours. **D2, claim and table:** abstract, method figure, main table, limitations; know the claim, its number and its scale; 30–45 minutes. **D1, one line:** abstract and the figure; know it exists and what it would change; 10 minutes.

Order is the order to read them in. A paper you have already worked with (TurboVLA, LeWM) still gets a full pass: working with code is not the same as having read the paper.

## Foundations (read first, D4)

1. **LeWorldModel: stable end-to-end JEPA from pixels** (2603.19312). D4. W1's recipe. Derive the SIGReg term and say why it prevents collapse without a target network. Know every number in the LeWM table that your G0 bar was built from.
2. **V-JEPA 2 / V-JEPA 2.1** (2603.14482). D4 for the objective and masking, D2 for the scaling results. The field's reference JEPA; Uma's World Models posting says "JEPA highly valued" with this lineage in mind.
3. **SmolVLA** (2506.01844). D4. The policy you run most; know the frozen-VLM decision, the flow-matching expert, the chunk size, and what its LIBERO numbers were measured with.
4. **Real-Time Chunking** (2506.07339). D3. The guidance mechanism you compared against in G13b; be able to say why exact guidance costs 2× and why prefix overwrite is gradient-free.
5. **The Planning Limits of Latent World Models** (2609.39235). D3. The horizon result that explained G1b's below-chance short rollouts; you will be asked "why did planning fail" and this is half the answer.
6. **D-JEPA** (2609.24749). D3. Read §3 slowly; you (we) misdescribed it once. Know the difference between adding a loss and training a relational operator on executed outcomes, and why your proxy-label version does not test the paper.
7. **No Free Checker: a survey of verifiers** (2609.09250). D3. The taxonomy behind your v1.5 tiers. Read §5 before writing any judge bar.

## Your own results' sources (D3)

8. **TurboVLA** (2607.27205). D3, then re-read §4–5.1 at D4 against upstream `turbovla/models/`. You ported it wrong; be able to list the four divergences and what Table 3 (language ablation) implies about the expected task-swap effect.
9. **VLA-JEPA** (2602.10098). D3. You fixed its future-frame leak upstream (#4381); know what the leak was and how the world-model loss is used at training time only.
10. **REKI** (2610.22178). D3. Register tokens, the oscillation metric you reused in G14, and what the Gated DeltaNet stack would cost to port to vla.cpp.
11. **Robometer** (RSS 2026, in KB). D3. What it was trained on (>1M trajectories, progress and success labels), why zero-shot transfer to SO-101 failed, what the G9b re-analysis showed about episode length.
12. **Flora-Rewarder** (NVIDIA, in KB). D2 for the model, D3 for the prompt and input protocol you copied in G-flora; the licence section too.
13. **LeWM / ContactGuard** (2608.13438). D3. The multi-view recipe and the event-anchored grasp check (W1-C1).
14. **FAIL-Detect** (2503.08558) and **SAFE** (2506.09937). D3 for the conformal construction, since your W1-C2 band was not conformal across time; be able to say exactly why (μ, σ fit on the same successes).
15. **ReAP** (2610.16309). D3. The risk head you trained; know their label definition so you can say how VLM-marked onsets differ from theirs.
16. **RoboJEPA** (2610.10515). D2 for method, D3 for the scaling-law section; it defines the scaling angle in §5.
17. **H-JEPA** (2610.06805). D2. Tested as G6b; know what the level-2 slow features were supposed to fix and why loops in joint space defeat them.
18. **One-step prediction fidelity is a poor proxy** (2610.19375). D3. Short paper, big consequence: r ≈ −0.01, and the matched continued-training control now in §7. Reproduce its argument in one paragraph.
19. **Physical-state grounding ×2** (2608.06799, 2609.03565). D2 each. The state-head angle; read with 18 in mind.
20. **HexaShell** (2610.23479). D2. The 739 s / 26 s number and why the agent stays out of the control loop.

## The harness papers (D2 each; D3 for 21)

21. **RegenHarness** (2609.27612). D3. Two-field results, idempotent dispatch, verifier isolation: the v1.5 contracts came from here.
22. **SEAR** (2610.18792), **Robo-COP** (2610.09228), **Mulligan** (2610.05882), **ReTrace** (2610.15814), **Trace-to-Architecture** (2610.16725), **Embodied Turing Machines** (2610.12369). D2 each. Know which v1.5 rule each one produced (the table in v1.5 §0 maps them).

## The prep queue (D3 each, before its bar is final; read the week its run is scheduled)

23. Smoothness ranks skill (2610.17427). Read §4 tables; the partial-correlation design is the whole paper.
24. State criticality (2610.17709). §3 estimator, Table 1.
25. RAFAIL (2609.18324). §3 graph construction, Table 2.
26. Robot-WATCHDOG (2610.13403). §3 and the real-robot section.
27. FailBench (2609.03611). §4 metric definitions and the per-model table; read ArmnetBench (2607.24481) §scoring first.
28. Measuring the stability assumption behind action chunking (2610.01626). §2 statistic, Fig. 2.
29. GeoAAC (2609.20776). §3.2, Table 3.
30. Fewer steps, better actions (2609.21216). §4 and the steps-vs-success table.
31. SMILE (2608.29432). §3, Fig. 4.
32. RESTA (2610.18163). §3 and the data-count ablation.
33. Quality-aware VLA pretraining (2610.18869). §5, Table 2; the design, not the labels.
34. P² progress probes (2610.31613). §IV–V.
35. GR-BC (2610.20504). The graph construction and the held-out question.
36. Coverage shift in VLM success monitoring (2610.24801). Short; the 31 → 50 % false-accept number and the experimental design.
37. VLA-Rolling (2610.25043) and PICES (2610.19656). D2 each until the robot day; D3 the week before it.

## For the Uma postings specifically (D2, read in the two weeks before an interview)

38. Grounding DINO (the fusion module TurboVLA borrows; needed to explain the port divergence properly).
39. UMI (universal manipulation interface) and one egocentric-data paper (EgoDex or Ego4D-based retargeting): the Data posting's "UMI-style demonstrations" and "egocentric recordings"; you have no hands-on here, so know the vocabulary and the failure modes.
40. LAPA or Genie-style latent actions: the "action inference from off-robot data" line in the Data posting.
41. One sim2real locomotion paper from the Optimus/humanoid lineage: the World Models posting mentions "sim2real RL for locomotion"; one paper is enough to ask a good question.
42. DINO-WM (2411.04983, in KB) and TD-JEPA: the two latent-planning references everyone at a world-models team will assume you know.

## Not papers, but read at D3

43. `journal/EXPERIMENTS.md`, `journal/LESSONS.md`, `journal/DECISIONS.md`: your own register, cover to cover, once, with this document open.
44. `projects/vla-jepa-world-model/latent-stack-architecture-v1.5.md` and `V1.0_REVIEW.md`: the design you would be asked to defend.
45. LeRobot's `LeRobotDataset` v3 format code and `policies/smolvla/modeling_smolvla.py`: you review PRs against these; read them as a maintainer would.

Total at the stated depths: roughly 45–55 hours. At four hours a week that is three months; at eight it is six weeks. Foundations (1–7) and your own sources (8–20) come first; the prep queue is read the week each paper runs; the interview block is read when there is an interview.
