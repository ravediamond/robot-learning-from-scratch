# Chapter 16. The interview

What a research-engineer interview at a world-models robotics company tests, question by question, with the answer that comes from this repository. Each answer is a template, not a script: the numbers and gate IDs are yours; the sentences should become yours by the time you have done the exercises in the earlier chapters.

## 16.1 What they are measuring

Four things, in rough order of weight for an RE role:

1. **Can you make a research idea run on real hardware and know whether it worked.** Everything in chapters 1–9.
2. **Do you understand the models well enough to debug them.** Chapters 11–12; the TurboVLA chain; the world-model roles.
3. **Can you build and own pipelines.** The Jetson image, vla.cpp ports, the harness, parity and strict-load checks; and the distributed-training vocabulary from chapter 14 said honestly.
4. **Will you be rigorous when nobody is watching.** The register, the correction rows, the verdict words.

Formal credentials (a PhD, a robotics degree) matter less than many postings suggest ("what matters most is what you've built and shipped"); they matter more if you cannot explain what you built. The exercises are the difference.

## 16.1a The shape of the loop, from public reports

What follows is assembled from public interview reports and coaching write-ups for research-engineer loops at frontier labs (Anthropic, DeepMind) and robot-learning companies (Figure, Skild, 1X), read on 10-23 (sources at the end of the chapter). Small robotics startups run a shorter version of the same thing; the rounds are the same, there are fewer of them, and the founders sit in.

| Stage | Typical form | What it tests | Where this book prepares it |
|---|---|---|---|
| Recruiter screen | 30 min | Fit, motivation, the two-minute story | 16.2 |
| Technical screen | 45–60 min, one engineer; at robot companies often *no* coding, "can you reason about embodied learning" | Vocabulary, one project in depth | 16.4, 16.5a |
| Take-home | 5–7 day window, sometimes a 3-hour timed one | Process: alternatives, trade-offs, stated limitations — graded on the write-up more than the number | 16.7a |
| Live ML coding | 45–60 min, PyTorch/NumPy, no autocomplete, no AI | Implement a component (attention, a training loop, a loss) and make it run | 16.4a |
| "Broken training script" | 45 min | Find subtle bugs by reasoning about what the model should be doing | 16.4a |
| Algorithmic coding | 2 × 20 min at DeepMind; a CoderPad problem at Skild; rare at small startups | Fluency; the code must run | 16.4a (last paragraph) |
| ML fundamentals | A quiz: regularisation, convexity, probability, ELBO | Whether the equations are yours | 16.4b |
| Paper discussion | 45 min on one of their papers or an adjacent one | Setup, claims, ablations; what would falsify it; the distinguishing experiment | Chapter 2a, 16.3a |
| Research discussion | 45 min | Taste and intuition; holding and defending a view under pushback | 16.7, 16.9 |
| Project deep-dive | 45–60 min; reported as the round that decides | Three levels deep on any number you have said | 16.2a |
| Systems design | 45 min | Training and inference infrastructure; for robots, the data and evaluation pipeline | 16.5 |
| Values / behavioural | 30–45 min | Disagreement, design-choice reasoning, honesty at the edge of what you know | 16.9a |

Two things the reports agree on. First, the gap between "research scientist" and "research engineer" has narrowed: the RE is expected to read papers and propose the next experiment, the RS to write code that runs. Second, interviewers deliberately probe the edge of what you know and push back; the behaviour they are scoring is whether you say "I don't know" cleanly and distinguish being wrong from being misunderstood. The method's verdict words are the same habit.

Rounds that appear in some loops and not in the table, each reported at least once:

- **Research presentation / job talk** (about 10 slides, defended like a thesis): 16.2b.
- **AI-assisted coding**: you get a small codebase and an LLM assistant; scored on how you direct it, not how much you type: 16.4a.
- **Live data-modelling scaffold**: a working but mediocre training script on a real dataset; find the planted bugs, do quick exploratory analysis (class imbalance, leakage, duplicates), improve it, and justify why each change should help: 16.4a.
- **Problem session**: an hour reasoning aloud through an open robot-learning or scaling problem: 16.5a.
- **Simulation debugging**: a small simulated robot with a deliberately broken controller; scored on how you instrument and diagnose (sign of a gain, contact model, frame conventions): 16.5a.

**Ask the recruiter what each round covers.** It is allowed, it is expected, and every candidate write-up that went badly includes "the format was not what I prepared for".

Timelines: 3–5 weeks at a startup, 6–10 weeks at a lab, longer with hiring-committee review. Reference checks may happen mid-loop. Ask at the screen which rounds the loop has; the answer tells you which of the sections below to rehearse.

## 16.1b How interviewers actually score, from the other side of the table

Interviewer write-ups agree on a short list, and almost none of it is "knew the answer". A candidate who finished a coding exercise was still rejected because *how* they solved it did not meet the bar (Eugene Yan). What is scored, in every round:

1. **Clarifying questions before solutions.** "The candidate's questions are more insightful on their ability than the eventual solution." Ask what the robot is, what the data is, what failure costs, what "done" means.
2. **The simplest thing first.** Strong candidates propose a baseline and say what failure of the baseline would justify something more complex; weak ones jump to the most sophisticated model they know. This is chapter 3 said out loud.
3. **Numbers, not adjectives.** Latency in ms, data in episodes, n per arm, memory per parameter, GPU-hours. "Abstract talky-talk without estimates" is a named red flag.
4. **Risks named before they are asked for.** Leakage (statistics on the full dataset, frame-level splits, tuning on test), metric blind spots ("I'd use this as a proxy for…; one risk with this metric is…"), edge cases (empty input, single class, NaNs). If you notice one late: "I should flag a potential issue here."
5. **Uncertainty handled honestly.** "It depends" is fine if you say on what. Overconfidence is penalised more than ignorance.
6. **Your contribution, exactly.** Interviewers probe to separate what you did from what the team (or the tools) did. Overstating and over-modesty are both red flags.
7. **Failures volunteered.** What went wrong, how you diagnosed it, what you changed. Candidates who only tell clean stories read as either inexperienced or evasive.
8. **Brevity.** Long-winded behavioural answers and deep-dive monologues are the most-reported self-inflicted failure. Answer in two minutes, then stop and let them steer.

For robot-learning roles, recruiter and hiring-manager guides add a domain layer:

| Red flag | Green flag |
|---|---|
| Claims that do not survive about **three layers of "why"** | Names versions, platforms and tools without prompting |
| Pure deep-learning background, no grounding in kinematics, actuator limits, sensor noise, latency | Knows where the real-time boundary sits and what runs on it |
| Vague answer to "tell me about a real failure and how you diagnosed it" | Describes what failed, what changed, what they would do differently |
| No awareness of the sim-to-real (or lab-to-field) gap | Thinks about data quality, not only architecture |
| A policy trained in a notebook, never run on hardware | Real-robot datasets uploaded, open-source code that runs, a commit history of problems solved |

The last row is the strongest asset of a one-person lab: public datasets, merged open-source work and a register with correction rows are exactly the evidence recruiters are told cannot be faked. Say where they are, with links, in the first five minutes.

## 16.1c What a frontier-lab founder says he hires for

Sergey Levine (co-founder, Physical Intelligence) was asked on stage, in a talk read in transcript on 10-27, whether a strong software engineer without a publication record but with the right mindset can be hired as a robotics researcher at PI. The answer was "yes, absolutely", and the rest of his answer is the most useful job description in this book, because it comes from the person who signs the offer:

1. **The bottleneck is signalling, not credentials.** "The other party needs to have some way to evaluate that, and that's very, very hard"; he wishes there were a better way. Read it from your side: the lab has no instrument for mindset, so the candidate has to build one. Everything in 16.1b's last row — public datasets, merged code, a register with correction rows — is that instrument. This section says what it should measure.
2. **The one trait: attention to detail, at every level of the stack.** His example: the model is not working, and the reason is not the learning rate or the dataset; it is a robot whose fingers leave a gap when commanded closed, because someone did not seat a connector. "You have to go there, plug in the connector properly, test it yourself, and then go back to training your models." A robotics researcher's day runs from tightening the gripper screws to training on thousands of GPUs, and the fault can sit anywhere in between.
3. **Specialising works against it.** Smart engineers "have a natural tendency to specialize", and that is "antithetical" to the trait he hires for. The culture he describes: everyone has a superpower, and a burning desire to understand the whole stack; the people who turned up to the actuator-and-sensor talk were the RL and infrastructure people.
4. **Mindset over record.** People with a top-school PhD and strong publications but not the mindset exist; so do people with the mindset, no credentials, and "the ability to get there in the right environment". His lesson for himself: do not prejudge, look at how someone approaches a problem, their actual thought process. Willingness to learn, and to say what you do not know, are named explicitly.

What this repository already has, mapped onto point 2. Each row is a fault that presented at one layer and lived at another; that is the shape his connector story has, and the shape an interviewer can verify from the register:

| Presented as | Lived at | Found by | Changed | Where |
|---|---|---|---|---|
| A world-model result (G11-A) | Checkpoint loading: `strict=False` plus a key remap that did not fit transformers 5.5.4 left the ViT encoder at random init | The Jetson strict-load check (192 missing / 192 unexpected encoder keys) | Gate voided (the 12.78 ms timing stands); strict load mandatory (chapter 6.2) | G11-A re-check (2026-10-18); post `two-gates-i-voided` |
| REKI's Orin latency, 318 ms (G12) | The benchmark itself: a random-init Qwen2.5-1.5B (no vision tower, no Gated DeltaNet) and a hand-written expert stood in for REKI | Its parts did not sum (135.4 + 230.3 ≠ 318.2 ms) | G12 void; the parts-must-sum rule | G12 re-check (2026-10-18) |
| "TurboVLA ignores the instruction", first blamed on our port | The data: the instruction is redundant on ~94 % of the bowls frames (the pose leaks the side), so upstream's own trainer on the same data reads it no better (task swap 0.020°) | Path A: the authors' own trainer, full LIBERO init, against a pre-registered bar | Port claim withdrawn; data claim stands; parity before diagnosis (chapter 8.4) | TurboVLA correction (2026-10-23); path A (2026-10-23) |
| One failed gate too many in the scoreboard count | The register file: unescaped `\|Δ\|` pipes in the vla.cpp determinism row shifted its cells and faked a fail | A hand count of the fails against the parser | Pipes escaped; the parser splits on unescaped pipes only; 73 gates, 20 failed | WS-039 log (2026-10-26) |
| A failure monitor that works (REAP v1, AUROC 0.93) | The features and the horizon: a time feature plus episode length carried the outcome (elapsed time alone caught 0/8) | REAP v2's controls: no time feature, common-horizon truncation (374 frames), shuffled and zero actions | AUROC 0.93 → 0.79; verdict not proven → fail | REAP monitors and REAP v2 (2026-10-23) |
| "LeRobot won't run on the Orin" | The platform: no channel ships a cp312 CUDA torch wheel for Jetson; five things built from source, six traps, 5 h 33 min of which PyTorch is 92 % | Building it | The published image and the merged Dockerfile (LeRobot #4599) | Post `lerobot-on-jetson-no-pip-install-torch` (2026-09-09); chapter 10.3 |
| A jump at every chunk boundary in vla.cpp | The engine: each chunk starts from fresh noise; ggml lacks backward rules for flash attention and concat, so RTC's guidance is not available | Measuring the jump on 254 boundaries per delay | Gradient-free prefix overwrite: jump cut 71–73 % at +0.4 ms per chunk | Post `vla-chunk-boundary-jumps` (2026-10-25); G13b in vla.cpp |

The full, public version of this list (12 faults, by layer) is the ledger of WS-044, `shissorobotics.com/faults/`.

Public at shissorobotics.com/faults/: 12 faults, 11 cross-layer.

Three things to do with the table.

- **Put it on the register, in public, by layer.** The scoreboard counts verdicts; it does not say *where the faults were*. A page that lists each correction with its presenting layer, its true layer and the probe that found it is the only artefact that answers "how do we evaluate attention to detail across the stack" without a referral. (Queued as a site item; see `journal/workstack/QUEUE.md`.)
- **Say the lowest layer in the first two minutes.** The 16.2 opener names gates and numbers; add one clause naming the lowest-level fault you fixed with your own hands (the connector-equivalent: the strict-load encoder, the pipe in the register, the Orin build). It is the sentence his hiring heuristic is listening for.
- **Do not describe yourself as "the model person" or "the eval person".** 16.9c has the row. The honest version of your profile is a specialism (evaluation, harnesses) plus evidence that you went down the stack every time the number did not make sense; the table is the evidence.

Two more things from the same talk belong in other sections, because they are the pushback a PI-style interviewer will give. On world models, his position is that "the most important thing for building foundation models, paradoxically, is not the model itself. It's the data", that the model class (video, world model, LLM, diffusion head) is "a downstream choice", and that the current focus on world models "is maybe a little bit distracting" from the data question — while also expecting the ultimate model to contain a world-model component for reasoning below the semantic level (rock climbing, not recipes). 16.4's "Why JEPA" answer has to survive that; 16.7b says how. On RL: deployment with human interventions teaches the robot to *fix* a mistake, not to *avoid* it; avoidance is where RL comes in, after autonomy makes the data; it works in his lab, not yet in deployment. 16.5a's RL question has the matching answer from this rig.

## 16.2 The opening: "Tell me about your robotics work"

Two minutes. Shape: the setup and the three conditions in one sentence; the scoreboard in one sentence; three results with numbers, each a different way a paper fails to transfer (the cheap baseline, the role, the budget); the corrections; the latest arm result. (Rewritten 2026-10-30 for the new thesis, chapter 15.4a: the old opener asked what a JEPA world model buys on cheap hardware; the register now covers seven families.)

> "I re-run published robot-learning claims on an SO-101 and a Jetson Orin NX — the hardware people can buy — with the cheapest baseline measured first and the pass mark committed to git before any number exists. Twenty-one papers so far, about $45 of rented compute, every result and every correction public. Four reproduce, twelve fail, three are not proven, one I withdrew. The fails are rarely "the method does not work"; they are "the cheap thing did as well" or "the budget is not there". The commanded joints rank action chunks better than a JEPA world model — 0.68 to 0.90 against 0.60 — yet the same model reads the task outcome at rest perfectly, AUROC 1.0: here it is a verifier, not a planner. Episode length beats a 4B reward model as a failure detector, 0.94 against 0.82. PACT-VLA's chunk selection replicates offline — 0.27 degrees closer to the demonstration than a random pick, interval clear of zero — but four candidates cost 37 control ticks on the Orin instead of 8, so the paper's "no added delay" fails. A gradient-free prefix overwrite I wrote cuts the chunk-boundary jump 71 to 73 percent for 0.4 milliseconds in vla.cpp. The corrections I am proudest of came from going down the stack: a `strict=False` load that left an encoder at random init, a register row whose unescaped pipes faked a fail, and a data loader that fed two models the red channel only — each caught by a check, each corrected with the old row kept. On the arm: REKI, trained on my data, put the bowls in 19 of 20 and 19 of 20 times."

The 16.1c clause ("came from going down the stack") stays: it is the only sentence about *how* you work rather than what you found. Drop the prefix-overwrite sentence if you are behind at 90 s. Do not quote the verdict counts without the date: they change every week (as of 2026-10-30, from the claims build). Pacing marks and the three-level ladder for every number: `exercises/interview/opener.md`.

## 16.2a The project deep-dive

The two-minute opener is the door; this round is the room. One engineer, 45–60 minutes, one project of your choosing, and the format is a chain of *why*: why that baseline, why that n, why that threshold, why not the other method, what did the number look like before you fixed the bug, what would you do with ten GPUs. Reports call it the round that decides, because it is the only one where nothing can be rehearsed as a sentence.

Prepare three levels for every number in 16.2. Level one is the number and its n ("0.60 AUROC vs 0.68–0.90, 54 episodes"). Level two is the mechanism (chapter 8's six-line form). Level three is the alternative you did not take and why ("I could have trained W1 on policy rollouts instead of teleop; I didn't because I had none; ContactGuard's result says that is probably where the monitor failure lives"). Interviewers go to level three on two or three numbers; if level three is missing on the one they pick, the rest of the story is discounted.

Choose the project by what they do. For a world-models company, the W1 chain (G0 → G1b → G10 → C1 → W2′) because it ends with a role, not a model. For a VLA company, the TurboVLA chain because it ends with a withdrawn claim and a reproduction. Have the second ready as the "another one" they will ask for.

**Structure** (the shape interviewer guides converge on): the problem and why it mattered → the approach and the architecture → the key decisions, each with the alternative you rejected → what went wrong and how you found it → the outcome and how you know it is real. Lead with the hard part and the failure, not the setup; the setup is one sentence.

**Follow-ups to rehearse**, by group: *context* (why this problem, who it was for, what you personally decided); *trade-offs* (why this over the alternative, would you still choose it, what changes at 10× scale); *debugging* (hardest bug, how you found it); *evaluation* (was it a real evaluation or vibes, what metric, what n, what baseline); *reflection* (what you would do differently, what you would try with more time).

**Mistakes reported most often**: monologuing without pausing (stop every two minutes and ask "want me to go deeper there or move on?"); naming technologies instead of decisions; being unable to go one level deeper on an implementation detail; choosing a project where you were not the main decision-maker. On the last: in this repository the tools wrote much of the code; say so, and say which decisions were yours (chapter 15.3, 16.9b).

Bring one figure you can draw on a whiteboard in a minute: the G1b decoy diagram (position control makes decoys differ in arm state), or the chunk-boundary jump. A drawn mechanism is remembered; a recited one is not.

## 16.2b The research presentation

Some loops replace or extend the deep-dive with a 30–45 minute talk on your work, about 10 slides, followed by questions in the style of a thesis defence. The outline that fits this repository:

1. The question, as a decision (one slide; chapter 1).
2. The rig and its budget: arm, cameras, edge computer, rented GPU, dollars spent (one slide; chapter 10).
3. The method in one picture: decision → baseline → pre-registered bar → run → verdict → register row (chapters 3–9).
4. Result 1 with its baseline and n: the world model learns dynamics (G0) but loses to commanded joints as a planner (G1b), with the mechanism (Planning Limits, position control).
5. Result 2: the monitor that did not transfer (W1-C, C1) and the action-sensitivity test that explains it (§28).
6. Result 3: the judge — length beat a 4B reward model; a deterministic judge with zero false positives (G9c-3, verifier card).
7. A correction told as a sequence: TurboVLA, the port divergence, path A for $2, the claim withdrawn (chapter 8).
8. Engineering: vla.cpp on the Orin, prefix overwrite at chunk boundaries, parity checks.
9. What would change the conclusions: rollouts with failures, object-state tasks, selection over policy samples — the data engine (the next experiments, pre-registered).
10. What you would do with their resources, in numbers.

Every slide carries a number with its n and its baseline; no slide carries an adjective without one. Rehearse it twice with a timer and once with someone hostile.

## 16.3 The method questions

*"How do you decide what to run?"* Chapter 1. The sentence; RoboJEPA cancelled; H0 ended in an afternoon.

*"How do you know a result is real?"* Chapters 3, 6, 7. Baseline first; instrument checks; episode bootstrap and paired CI; the n sentence.

*"Pre-registration, really?"* Chapter 4. The commit hash; G9c-3 as the honest exception.

*"Fail vs not proven?"* Chapter 5. The checklist; REAP and C1.

*"A result you changed your mind about."* Chapter 9. Any correction row, told as a sequence.

## 16.3a The paper-discussion round

They send a paper the day before, or hand you one in the room with twenty minutes. The questions are nearly fixed across companies:

1. "Summarise the setup and the central claim." (The one sentence of 2a.2: subject, comparison, number, unit, n, platform.)
2. "What do the ablations show, and which one is missing?" (2a.3: the matched control, the cheapest baseline, the quoted row.)
3. "What is the most plausible alternative explanation for the main result?" (Your confound list: length, policy identity, time, threshold chosen on the test sweep, the proxy with no bridge.)
4. "What would falsify the central hypothesis?" (2a.4 step 3: the cheapest falsifying measurement.)
5. "Which single experiment most efficiently distinguishes the authors' hypothesis from your alternative?" (One arm, one control, one number.)
6. "What would a replication look like, and what would you expect to break first on our hardware?" (The translation table.)
7. "If you were designing the follow-up, what would you test and why?" (Step 2 of the ladder: what it would change.)

The order to read in the room is 2a.2's: abstract, main table, setup, method, limitations, introduction last; twenty minutes is enough for the first four. Say which sections you did not get to. Candidates lose this round by summarising the introduction, by praising the paper, and by proposing a follow-up that is a bigger version of the same experiment rather than one that could make it false.

Practice set, from the knowledge base, each with its answer already in an "a" chapter: ContactGuard (12a.8 — the missing shuffled-chunk control and the teleop-vs-rollout confound), 2610.19375 (12c — why the control absorbed the gain), D-JEPA (12b — the executed-outcome supervision nobody can afford on a real robot), REKI (11a — the frozen-encoder confound), CUPID (13b — influence vs quality, the four-arm test), SALT (12c — worse one-step error, better policy). Do one a day from the PDF, write the seven answers in the 2a.8 template, then check against the chapter.

If they hand you their own paper: the same questions, but the follow-up you propose should be one you could run in your first month there, on their hardware, with their data. Say its cost.

## 16.4 The model questions

*"Why JEPA and not a video world model?"* Chapter 12d.6's last paragraph, with LaLa-World's matched numbers (59.4 vs 48.0, 6× cheaper, 13.9× faster) and Fast-WAM's finding that test-time imagination is not what makes video models useful.

*"Why did planning fail?"* Chapter 12.4.1 and chapter 3.3.1. Position control; decoys differ in arm state; p^7; the 0.133 ceiling; what G7 changes.

*"Why does the readout work and the monitor not?"* 12.4.3. Outcome is visible in the final frame; the surprise distribution widens on fresh data; imagination adds noise to a grasp check the encoder already answers.

*"What is SIGReg / EMA / VICReg for?"* 12.2. Collapse, three fixes, effective rank 5.8 → 24–30.

*"Explain flow matching to me."* 11.3. The path, the velocity target, the Euler sampler, why shared noise for parity, why N controls latency.

*"Why did TurboVLA fail?"* Chapter 8. Six probes with numbers to fact 7; the generalisation; then the port divergence; then path A (0.020° vs 0.105) closing it: the data, not the model. Say what you would do differently (parity before diagnosis) and what the paper's own Table 3 says the architecture can do when the data forces it.

*"What's a chunk boundary and why does it matter?"* 11.4. d ≈ 7 steps of inference delay; fresh noise; the jump; prefix overwrite vs RTC on 254 boundaries.

## 16.4a The coding rounds

**Live ML coding.** One component, 45 minutes, PyTorch, an empty editor, no autocomplete. The reported prompts: multi-head self-attention from scratch; a training loop with a loss and an optimiser step; gradient descent as a function; a loss function from a paper. For this role, add the ones from your own stack: a flow-matching training step (sample t, interpolate x_t, predict velocity, MSE to x_1 − x_0) and its Euler sampler; a SIGReg term (project to random directions, compare the sketch to N(0,1)); the episode bootstrap for a paired CI; a chunked action executor with the prefix overwrite. Each is under 40 lines. Write every one of them cold, this week, time yourself, run it on a toy tensor, and keep the file in the repo (`journal/book/exercises/coding/`). The scoring rubric is: does it run, are the shapes right first time, did you test it on a tiny input before declaring done, did you narrate the shapes as you typed. Silence while typing is the common failure.

**The broken training script.** You are given training code that runs and converges to something wrong. The bugs are the ones you have met: normalisation statistics computed on the wrong split or missing at inference; `model.eval()` never called; a target that is not detached (the collapse of 12.2); the data loader not shuffling; a learning rate ten times off; an off-by-one that makes the chunk target include the current action; dtype casting that drops precision in the loss; a seed set after the data split; a leaked feature (REAP v1's elapsed time). Have a fixed order: (1) print shapes and dtypes at every boundary; (2) overfit one batch — if the loss does not go to zero the bug is in the loss or the target; (3) check the data pipeline against one hand-computed example; (4) check eval mode, normalisation, detach; (5) only then the hyper-parameters. Chapter 6 is this round as a method; strict load and parity are its two biggest items.

**How the debugging round goes wrong, from a candidate's own write-up** (OpenAI, a "general coding" slot that turned out to be a debugging session): nerves, then auditing the whole codebase instead of forming a hypothesis about where the bug lives; the interviewer had to say "the rest of the code is fine". The lessons are the drill's rules: ask about scope in the first minute; state a hypothesis before reading code; test it with one targeted print or assert; time-box each hypothesis to five minutes; watch for the interviewer's cues. A practitioner who sat several research-engineering loops reports the other common failure: **running out of time looking up syntax**. Practise the coding list with no documentation and no assistant until the APIs of chapter P2 are in your fingers.

**The AI-assisted coding round.** A small codebase, an LLM assistant in the editor, a task (fix the failing tests, then improve). Reported scoring: a clear plan before prompting; precise prompts; reading errors yourself; saying why you accept or reject each suggestion; a naive working solution and a benchmark before optimising. It rewards exactly the division of labour of chapter 15: the tool writes, you decide and verify. Say what you are checking in each suggestion.

**The data-modelling scaffold.** A working script on a real dataset, mediocre on purpose. Order: run it and read the metric; quick exploratory checks (label balance, duplicates across splits, a feature that leaks the label, units and normalisation); fix the planted bugs; one change at a time with a reason each ("this should help because…", then show it did). Interviewers who run this format call it the highest-signal round they have; you cannot memorise it, only practise the habit.

**Algorithmic coding.** Two 20-minute problems at DeepMind (trie, grid distance, linked lists); one CoderPad problem at Skild (in-place matrix inversion with complexity discussion); usually none at a ten-person startup. Two weeks of the standard medium set is enough insurance; do not let it displace the ML rounds, which are weighted higher for this role.

## 16.4b The fundamentals quiz

Reported questions, with where the answer lives:

- *L1 vs L2 regularisation, and the gradient intuition.* L1's gradient is constant in magnitude, so it drives small weights to exactly zero; L2's shrinks in proportion, so it never does. Sparse features → L1.
- *What is a convex function and why does it matter?* Any local minimum is global; neural losses are not convex, which is why seeds matter (chapter 7's 26-pp floor).
- *Derive the ELBO.* log p(x) ≥ E_q[log p(x|z)] − KL(q(z|x) ‖ p(z)). Know it because the JEPA argument (12.1) is that you do *not* want the reconstruction term.
- *Why EMA targets / why a stop-gradient?* Collapse (12.2). Say what collapsed in your own run (effective rank 5.8) and what fixed it.
- *Bias–variance; overfitting; how you would know.* Held-out split by episode, never by frame (chapter 7).
- *Bayes with coloured balls.* Write the posterior; it is a two-line calculation and they want to see you write it rather than reason in words.
- *Why bf16, not fp16; what breaks.* Range vs precision; loss scaling; where the master weights live (14.3).
- *Explain a confidence interval on a success rate.* Wilson, not normal, at n = 20 (chapter 7). Say what 7/10 means.
- *Why predict a chunk of actions rather than one?* Inference delay, temporal consistency, multimodality averaging (11.4).

If asked something you do not know, say so in one sentence and offer the nearest thing you do know. The reports are explicit that bluffing is scored and detected.

## 16.5 The pipeline and scale questions

*"Walk me through getting LeRobot running on a Jetson."* The five source builds, the six traps, 5 h 33 of which 92 % is PyTorch, the published image, the merged Dockerfile. Chapter 10.3 and the blog post.

*"How would you add a new policy to LeRobot, and how would you serve it from a GPU across the network?"* Chapter 14b: the three load-bearing names, the processors as the reproducibility trap, the plugin prefix, the remote-inference buffer and its starvation behaviour; then what changes when the policy is a vendor API (version pinning, latency distribution, evaluation still yours).

*"How would you port a model to a C++ engine and trust it?"* 14.1.2 and chapter 6. The seven mechanical sites; ggml's missing backward rules; `vla_predict_check` with shared noise; 0.80° vs 1°.

*"Have you trained at scale?"* 14.3.8, verbatim honesty. Then the vocabulary: effective batch, bf16, DDP vs FSDP, what breaks at 8 GPUs, profiler first, pilot-then-budget.

*"Design an evaluation pipeline for a fleet of robots."* The harness, generalised: idempotent dispatch, two-field results, tiered verifiers with report cards, sealed layouts, episode-level records, a register with correction rows, a weekly audit. Then the number: HexaShell's 739 s / 26 s, and why the agent stays out of the loop.

## 16.5a The embodied-learning reasoning questions

Robot-learning companies report a first technical round with no code, built on questions like these (the Figure set below is a public synthesis from their role descriptions, not leaked questions; the shape is right). Each has an answer in this repository.

*"Behaviour cloning on teleoperation data has a well-known failure mode."* Compounding error under covariate shift: the policy's own small deviations take it to states the demonstrations never covered. The fixes, in the order you have evidence for: more and more varied data (fact 7's pose shortcut is the data side of this); corrections and recoveries in the data (SEAR's point about the harness's own resets avoiding hard states); action chunking to reduce the number of decision points; a judge that stops the episode before the deviation becomes damage (G9c-3); residual or RL on top (14a) once the base is good enough to produce successes.

*"Why predict a sequence of actions instead of the next one?"* 11.4, then the cost: the chunk boundary, d ≈ 7 steps, the jump, and what RTC and your prefix overwrite do about it.

*"Why a slow large model and a fast small one, not one network?"* Latency budget: the fast loop must run at control rate on the robot; the slow one can run at 1–2 Hz and set the context. Your version is W1 as a slow verifier at rest and the ACT/SmolVLA loop at 30 Hz; the honest caveat is that on an Orin NX both have to fit in 16 GB.

*"You have 10 hours of demonstrations and budget for 50 more. What do you collect?"* 16.6 and §24: failed layouts first, measured by a selected-vs-random test at matched budget; recoveries; the frames where the instruction actually matters (fact 7 says 94 % of yours did not need it); never into the policy from failures. Then the n sentence: 50 hours is ~ 3,000 episodes, enough to resolve a 10-point difference; 10 hours is not.

*"How do you evaluate a manipulation policy when every trial costs robot time and every failure has physical consequences?"* Chapters 4 and 7 as one answer: pre-registered bar; paired layouts so noise cancels; sealed layouts; a Wilson interval at the n the day affords, said before the day; offline proxies (ranking AUROC, judge report cards) to decide *which* policies deserve robot time, with 2610.19375 as the warning that the proxy must be bridged; the judge runs only at rest and never alone for training data.

*"You ship a policy to 300 robots. It works in the lab and degrades in the field. Debug it."* Instrument first (which checkpoint, which normalisation statistics, which camera exposure); then coverage shift — the field states the lab never showed, and the judge's acceptance rate changing with them; then per-site paired comparison against the lab layout; then the data collected *from* the field, failures labelled, before any retraining. Robo-COP (14a) for the fleet-wide failure taxonomy; the harness's two-field results and correction rows for how the fleet's register would look.

*"Where does RL fit on top of imitation?"* 14a: residual RL on a frozen base once success is non-zero; needs a reward, and your evidence is that the only reliable reward on this rig is the deterministic judge at rest, not a learned one (Robometer vs length). Say what you have not done (no RL on hardware) before saying what you would do. If the interviewer is from a PI-style lab, the sequence they have in mind (16.1c) is: deploy under human interventions → the interventions are the data that fixes mistakes → autonomy rises → the robot's own experience plus RL removes the clumsiness. Your rig has the first two steps in miniature: corrections in the tidy-loop teleop set, failures labelled and kept out of the policy, the harness's two-field result that records *why* a run stopped. Say that the missing piece on your side is the reward (the judge runs at rest, not at control rate), and that the learned judges did not hold here: Robometer-4B failed as a zero-shot judge (G9a-AB complete: pooled AUROC 0.818 against a 0.85 bar, episode length alone 0.938), and the VLM judge fitted for G16 caught 0/8 failures at its threshold (Judge τ, 2026-10-19).

*"What's the robotics analogue of prompt expansion?"* (A question from the same talk: PI's zero-shot generalisation in π0.7 came from richer annotations on robot examples, by the same principle that makes prompt expansion work for video models — a more detailed prompt has higher mutual information with the output — but video-style captions are useless to a robot; the annotation has to describe the behaviour.) Your answer from the data side: fact 7's 94 % of frames where the instruction did not matter is the measurement that the current labels carry almost no information about the action; the §24 selected-vs-random test is how you would check that a richer label does. Then the honest limit: you have not trained with sub-task annotations; the π0.5-style decomposition ("pick up the towel, wipe the spill") is the first thing you would add, with the chunk-boundary cost (11.4) said.

*"A paper reports pretraining on human video lifted zero-shot success from 9 to 56 %. What do you make of it?"* Chapter 2a on a number: what is n per cell, what is the matched control (same extra training steps without the video — the 2610.19375 pattern), what is "unseen", who labelled success. Then the translation: VLA-JEPA's "w/o human video" row is the kind of control that would make it an evidence claim; Ego4WAM (13b) is where your own egocentric video would enter (§25).

*"Design the safety architecture for a learned policy near people."* Tiers: a reflex layer the policy cannot override (joint limits, torque/thermal, workspace fence — the Orin's thermal check and stillness rule in AGENTS.md are the small version); a monitor at control rate with a known false-alarm rate; a judge at rest; a human with a kill switch whose decisions are logged. Say that your W1 monitor failed the fresh-rollout transfer test and therefore runs in shadow mode only — that is the architecture decision they are asking about.

*Problem-session prompts reported or announced by robot-learning companies* (Skild's candidate reports list them as the shape to expect, not confirmed questions):

- *"You have 10 000 hours of teleoperation and 100 hours of real rollouts. One policy. How do you combine them?"* Rollouts are on-distribution for the policy and contain its failures; teleop is broad and clean. Weight rollouts up, label outcomes, never imitate failed rollouts (use them for the critic or the world model), and test the mixture at fixed compute with a held-out real-robot block (chapter 13b; the Data-role prep sheet).
- *"Why does behaviour cloning compound errors, and what does DAgger fix and not fix?"* DAgger queries an expert on the states the learner visits and adds those labels, which removes the covariate shift; it needs an expert available online (costly on a real robot), and it does not fix multimodality or a bad action representation.
- *"When would you learn a world model instead of a policy?"* When you need to evaluate or select actions you have not demonstrated, when you have far more unlabelled video than actions, or when you need a monitor or a simulator for evaluation; not when commanded actions already explain the outcome (your G1b). Say the result you have, then the conditions under which it would change (the data-engine use case).
- *"Sparse rewards: what before reward shaping?"* Demonstrations to seed, a success detector you trust (the verifier tiers of chapter 13), hindsight relabelling, residual RL on a competent base policy (14a).
- *"Make a training run resumable after a node dies."* Checkpoint model, optimiser, scheduler, sampler epoch and RNG states on a fixed cadence, atomically (write then rename), keep the last two; resume and verify the loss curve continues (P2.12).

*Embodiment grounding* (the red flag "deep learning without kinematics"). Have short answers for: forward vs inverse kinematics and why IK can have zero or many solutions; position control vs torque control and what your STS3215 servos can and cannot sense; PD control and what a wrong-sign gain does (it pushes away from the target and the joint runs to its limit); control rate vs policy rate and where the inference delay goes (chunk boundaries, chapter 11.4); camera latency and calibration drift; thermal limits on hobby servos. Chapter 10a explains each of these from zero with runnable examples (and its 10a.10 lists ten one-minute questions); chapter 10 has the numbers for your rig. In a simulation-debugging round, instrument first (plot command vs measured joint, check units and frame conventions), then hypothesise.

## 16.6 The data questions (for the Data posting)

*"Which data improves a policy?"* Chapter 13.3. Outcome-labelled rollouts first; corrections for recovery; instruction-dependent frames early; trim idle starts; failures never into the policy. The tests: curation signals as judges beyond length, then as training weights on a pooled gate with enough n.

*"What would you collect next?"* Failed layouts first with the selected-vs-random test; the deployed policy's own rollouts, failures included, for the world model (every monitor so far was trained on teleop only); counterfactual outcomes for every candidate (the D-JEPA supervision).

*"Egocentric / UMI / retargeting?"* Honest: no hands-on. Then the vocabulary from the reading list (hand pose → end-effector → joints via IK; viewpoint alignment; latent actions when no action labels exist) and the one question you would ask them about their pipeline.

## 16.7 The "what would you do here" questions

*"Reproduce paper X in two weeks: plan?"* Day 1: decision sentence, translation table, cheap baseline, protocol with bar, commit. Days 2–3: reference code running on their data at a smoke scale; parity against the released checkpoint; strict load. Days 4–8: the run with a pilot-measured budget. Days 9–10: episode-bootstrapped results, paired against the baseline, the confound checklist, the failure read if it failed. Days 11–14: write-up with the six-line mechanism form; register row; what it changes. Say which steps you would cut if the two weeks became one (the parity and the baseline are not the ones).

*"Our world model's loss is going down but the robot isn't improving."* 2610.19375 (r ≈ −0.01); check the instrument (what checkpoint is actually loaded on the robot); check whether the evaluation task has the property the model improves; compute the commanded-joint baseline on their ranking task; ask what the model is used *for* on the robot (planner, monitor, verifier) and test that role directly.

*"Our VLM judge disagrees with our operators 10 % of the time."* Which direction (FP vs FN); whole-episode or snapshot input; coverage shift as the policy improved; a report card against human labels per task; pair it with a tier-2 check at rest; never let it decide alone for training data.

## 16.7a The take-home

Five to seven days, sometimes a three-hour timed version. Reported forms: an open-ended implementation; a short empirical study on a provided dataset; a paper implementation with an extension. Reported grading: the process — alternatives considered, trade-offs documented, limitations stated — counts more than the headline number. This is the register in miniature, and the shape of a protocol file is the shape of a winning submission.

Deliver, in this order, in one README: the decision sentence (what question the work answers); the cheapest baseline and its number, run first; the protocol (what was fixed, what varied, the bar you set before running — commit it with a timestamp even in a take-home, it is the most distinctive thing you can do); the result with intervals and the n sentence; the confound list with what each would do to the verdict; what you did not do and why (time, compute, data); the one experiment you would run next with its cost. Then the code, runnable with one command, with a smoke test that finishes in a minute.

Time-box: a third of the window on getting a baseline and the pipeline to run end to end, a third on the method, a third on the write-up. Candidates lose by spending the whole window on the method and writing the README in the last hour. If something does not work, say so in the README with the mechanism; a *fail* with a mechanism scores above a *pass* without one.

For the three-hour timed version: baseline in the first 45 minutes, one change, the write-up as you go.

## 16.7b Holding the world-model view under the data pushback

The research-discussion round scores whether you can hold a view when the interviewer pushes on it (16.1a). The push you should expect from a PI-trained interviewer is Levine's (16.1c): the data trajectory is the foundation-model decision; the model class is downstream; the world-model focus distracts from it. Three mistakes, then the answer.

Mistakes: (1) defending JEPA as an architecture — you lose, because the claim is not about architecture; (2) conceding everything — then you have no view; (3) quoting LeCun — that is a philosophical debate and he will say so.

The answer that survives, in order:

1. *Agree with the premise, in his terms, from your register.* Every result that mattered on this rig was decided by what the data contained, not by the model: G1b (an action-only baseline ranks the chunks better than the world model: top-1 0.60 vs 0.68–0.90); C1 (policy identity alone, AUROC 0.76, beat every latent probe: the failures belonged to one policy); the TurboVLA chain (the instruction was redundant on ~94 % of the frames, and the authors' own trainer did no better on them). Those are data findings dressed as model gates — G1b and C1 by their baselines, TurboVLA by its correction row.
2. *Then the narrower claim you do hold.* On a GPU-poor rig the question is not "which model" but "which *role* the world model earns given the data you can afford to collect". Your evidence: verifier at rest, yes (G10, AUROC 1.0); planner, no (G1b); monitor, not proven (C1) and then failed (W1-A, REAP v2). That is a data-conditioned answer, which is his frame.
3. *Where his own position needs a world model.* He expects the ultimate model to contain one for reasoning below the semantic level — the rock-climbing example. On a low-cost rig the sub-semantic reasoning that matters is contact and object state, which is exactly where your next gates (object-state, G7) sit, and the cost of a JEPA predictor versus a video model is the reason it is the one you can afford to test (12d.6, the one matched study, LaLa-World: planning 59.4 vs 48.0 after six times less training, 13.9× faster, in simulation).
4. *The question back.* "What is the world model's role on your robot today — planner, monitor, verifier, data filter — and which experiment decided it?" (16.8, first bullet.) It turns the pushback into a comparison of evidence, which is the only exchange you can win.

Say step 1 before step 2; the order is the whole trick.

## 16.8 Questions to ask them

Pick three. They signal what you care about and they get you information you need.

- "What is the current role of the world model on the robot: planner, monitor, verifier, or data filter? What is the evidence for that choice?"
- "How are evaluation rollouts labelled, by whom, and what is the measured disagreement rate?"
- "What does a pre-registration look like here, and who signs off a bar before a run?"
- "What is the cheapest baseline your latent planner is compared against on the real robot?"
- "What broke the last time a training run went from 1 GPU to 8?"
- "Which result in the last quarter did the team change its mind about, and how was that recorded?"
- "What was the last fault that presented as a model problem and lived somewhere else in the stack, and who found it?" (16.1c; it also tells you whether the culture he describes is theirs.)

## 16.9 The negative-result question

They will ask about a failure. The shape that reads as competence: the bar, the number, the mechanism, the generalisation, what it changed, and one sentence on what you would have done differently. Never: "it didn't work"; never a list of things tried without a mechanism. The TurboVLA story and the G1b story are the two to have ready; the second is shorter. A third, new since 2026-10-30 and the shortest: **PACT-VLA** — bar (selection beats random and stale selection, and the cycle adds no delay), number (−0.27°, CI −0.44 to −0.09; 37 ticks vs 8), mechanism (vla.cpp samples four candidates sequentially and vision is 151 of 238 ms), generalisation (an inference-time method is a claim about a latency budget), what it changed (every inference-time protocol now times the cycle on the Orin first), what you would do differently (time the cycle before running the quality gate, then decide whether the quality gate is still worth an evening; here it was).

## 16.9a The behavioural round

Reported questions: a time you disagreed with a colleague and how it ended; the reasoning behind a key design choice; a project you are proud of; why this company. The scoring is for specificity and for whether you can describe the other side's position fairly.

Have these three ready, each with the other side stated first:

*Disagreement.* The §18/§20 protocols: the tool on the Orin pushed back that the offline steps as written asked for success deltas no recording could give; it was right; the protocols were rewritten as proxies plus a closed-loop block on robot day. Tell it as "I was wrong, here is how I found out, here is what changed in the process" (rule 2 of chapter 15 came from it). If they want a human disagreement, open-source code review gives you one; pick a case where the author's position was reasonable and say why.

*Design choice.* Prefix overwrite rather than RTC in vla.cpp: gradient-free because ggml had no backward rules, 0.4 ms, 71–73 % of the jump; the cost is that it is not RTC and you have to say what RTC would buy (the soft-mask comparison now in NEXT_ANGLES). Or: making the judge deterministic rather than learned, after length beat the 4B reward model.

*Why here.* Specific to their published work, in one sentence, with one question you want their data to answer that yours cannot. Not "world models are the future".

On honesty: the reports say interviewers push past what you know on purpose. The expected response is a clean "I don't know that; what I do know is …" — the same discipline as *not proven*. Do not negotiate with the pushback; answer it or concede it.

## 16.9b Talking about the AI tools you used

This repository was built with coding agents doing much of the typing. Interviewers will find out, and the question is not whether you used them but whether you can show what you decided. The defensible position, stated before they ask:

- What the tools did: wrote code, ran scripts, drafted protocols and reports.
- What you did: chose each question, read the paper's method section, wrote or rewrote every bar, read each failure for a mechanism, caught the errors (the random-init encoder, the stand-in model, the TurboVLA port, the protocols that asked for impossible deltas), signed each verdict (chapter 15.3).
- The evidence: the register's correction rows, the rule that a tool-written bar is rewritten by you before a run, and the live-coding round, where you write the same components with no assistant.

Never let a tool-written number reach a slide you cannot derive. The project deep-dive's third "why" is where that shows.

## 16.9c Mistakes that cost offers, and their fixes

| Mistake | Fix |
|---|---|
| Generic best practice ("try a bigger model, tune hyper-parameters") | Clarify the context, start from a baseline, say what would justify more complexity |
| Naming a metric without saying what it misses | "I'd use X as a proxy for Y; one risk with it is Z" |
| Leakage unmentioned until the interviewer finds it | Name one leakage risk in every data answer, unprompted |
| Long answers | Two minutes, then a check-in question |
| Clean success stories only | Lead with the hard part and the failure |
| Bluffing at the edge of knowledge | "I don't know; the nearest thing I know is…" |
| One or two wrong fundamentals answers after strong coding | Reported as enough to reject; drill P1's exercises and 16.4b until they are reflexes |
| Googling syntax in the coding round | Practise the list with no docs (16.4a) |
| Unprepared for the round's format | Ask the recruiter what each round covers |
| "Why you, not anyone else?" answered generically | One sentence tying your evidence (public datasets, open-source work, the register) to their posting's first bullet |
| Introducing yourself as "the model person" / "the eval person" | Name the specialism, then the lowest layer you debugged yourself (16.1c's table); specialising is the named red flag of the one founder on record |

## 16.10 Final exercise

Record yourself answering 16.2 and every question in 16.3–16.7 without notes. Listen back. Mark every place you said a number you could not derive, a gate you could not describe, or a word (*fail*, *pass*, *significant*) you could not justify. Those are the chapters to redo. When the recording has none, you are ready.

Then the rounds that cannot be recorded. A four-week plan that matches the loop:

- **Week 1:** 16.1b, 16.1c and 16.9c read twice; the 16.2 opener with the 16.1c clause and 16.7b's dialogue rehearsed aloud (`exercises/interview/`); the 10-slide talk (16.2b) drafted; coding items 14, 23, 02, 03 cold.
- **Week 2:** the deep-dive rehearsed with a friend who asks "why" three times per answer; the paper-discussion practice set (16.3a), one a day; coding items 15, 11–13, 08.
- **Week 3:** the debugging drill and one data-modelling scaffold on a public dataset (16.4a); the problem-session prompts (16.5a) answered aloud in 5 minutes each; the embodiment-grounding list.
- **Week 4:** a full mock loop in one day — screen, coding, deep-dive, paper, behavioural — with breaks of the real length; then fix the three weakest answers.

The older checklist still holds: write the five coding components of 16.4a cold and time them; do the 16.3a practice set, one paper a day, from the PDF; have someone hand you an unread paper and twenty minutes and run the seven questions; and do one three-hour mock take-home on a public dataset with the 16.7a README shape.

## Sources for 16.1a–16.9c

Added 10-27 for 16.1c, 16.5a (RL sequence, prompt-expansion analogue), 16.7b and 16.9c (primary source, a public talk, read in transcript): Sergey Levine, "How close are we to robots that can do anything?", on-stage interview (Physical Intelligence; π0 → π0.5 → π0.7, the data thesis, hiring without credentials, attention to detail across the stack). Quotations in 16.1c are from the transcript; the "neural thickets" paper he recommended (first author not named on stage) is listed in Appendix R as unverified.


Public reports and coaching syntheses, read 10-23; none is official company guidance, and company-specific question lists marked as "synthesised from role descriptions" are shapes, not leaks:

- Anthropic research-engineer loop (recruiter → technical screen → 5–7-day take-home → 4–5 h onsite with ML coding/debugging, systems design, research and paper discussion, values; the paper-discussion questions quoted in 16.3a): [Sundeep Teki, 2026](https://www.sundeepteki.org/advice/anthropic-research-engineer-interview-2026).
- DeepMind research-engineer loop (two algorithmic coding rounds, ML fundamentals quiz, ML design, team-lead and behavioural rounds; the fundamentals questions in 16.4b): [IGotAnOffer](https://igotanoffer.com/en/advice/google-deepmind-research-engineer-interview); process stage frequencies: [Glassdoor, DeepMind](https://www.glassdoor.co.uk/Interview/Google-DeepMind-Interview-Questions-E1596815.htm).
- Skild AI reports (CoderPad live coding; "write multi-headed self-attention in PyTorch without autocomplete or AI"): [Glassdoor, Skild AI](https://www.glassdoor.com/Interview/Skild-AI-Interview-Questions-E10127923.htm).
- Figure AI embodied-reasoning question set and loop shape (no-code first technical round; project deep-dive as the deciding round): [ombharatiya/AI-Engineer-Interview-Questions](https://github.com/ombharatiya/AI-Engineer-Interview-Questions/blob/main/14-company-interview-questions/figure-ai.md); [Glassdoor, Figure AI](https://www.glassdoor.com/Interview/Figure-AI-Interview-Questions-E9642582.htm).
- 1X AI-engineer loop shape (four rounds, 3–5 weeks; generic ML questions): [Dataford](https://dataford.io/interview-guides/1x-technologies-as/ai-engineer).

Added 10-24 for 16.1b, 16.2a–b, 16.4a, 16.5a, 16.9b–c (interviewer-side write-ups and candidate reports; robotics-specific lists are recruiter guides or candidate-reported shapes, not official processes):

- Interviewer side, what each round scores, red and green flags ("how they solved it did not meet the bar"): [Eugene Yan, How to Interview and Hire ML/AI Engineers](https://eugeneyan.com/writing/how-to-interview/); [Modern Descartes, Interviewing for ML/AI Engineers](https://www.moderndescartes.com/essays/ml_eng_interviewing/) (data-modelling scaffold as the highest-signal round; numbers vs "talky-talk").
- Candidate side, research-engineering loops, the research presentation, AI-assisted coding, ML coding failures (syntax lookup), fundamentals rejections, long-winded answers: [Yuan Meng, MLE Interview 2.0: Research Engineering and Scary Rounds](https://www.yuan-meng.com/posts/mle_interviews_2.0/).
- Project deep-dive structure, follow-up groups and common mistakes: [alexeygrigorev, AI engineering field guide, project deep dive](https://github.com/alexeygrigorev/ai-engineering-field-guide/blob/main/interview/questions/03-project-deep-dive.md).
- A debugging round that went wrong, and its lessons: [PracHub, OpenAI MLE interview experience](https://prachub.com/interview-experiences/openai-machine-learning-engineer-interview-experience-a-malloc-phone-screen-and-a-surprise-debugging-round).
- Mistakes that cost offers (generic answers, metrics without caveats, leakage, overconfidence): [InterviewNode](https://interviewnode.com/post/mistakes-that-cost-you-ml-interview-offers-and-how-to-fix-them).
- Robot-learning hiring signals and red flags (three layers of why, embodiment grounding, open-source and real-robot datasets as competence filters, simulation-debugging format): [HeroHunt, Recruiting robotics engineers for physical AI (2026)](https://www.herohunt.ai/blog/recruiting-robotics-engineers-for-physical-ai-2026/); [KORE1, How to hire robotics engineers in 2026](https://www.kore1.com/hire-robotics-engineers-2026/).
- Skild AI loop shape and problem-session prompts (candidate-reported; "treat it as the shape, not a contract"): [techinterview.org, How Skild AI interviews robot-learning and infra engineers](https://www.techinterview.org/post/3233477142/skild-ai-interview-guide/).
