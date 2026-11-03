# Evaluating Robot Learning on Cheap Hardware: a research engineer's method for a one-person lab

## How to use this book

This is the long form of `journal/METHOD.md`. That file is the map; this is the territory. It is written for a one-person lab: one engineer, one low-cost arm (an SO-101), an edge computer (a Jetson Orin NX), a laptop and rented GPUs, a few evenings and robot days a month. Its purpose is that the reader can run research-grade experiments with those means, know whether each one worked, and explain — to a colleague, a reviewer or an interview panel — what was done, why it was done that way, what the alternatives were, and what would change with ten times the resources.

The standard it aims at is "experienced research engineer": someone who has run enough experiments to have been fooled by most of the usual things, and who can therefore explain, for each rule, the specific failure the rule prevents. Rules without their failures are superstition. This book tries never to state a rule without its failure.

Every example comes from one real repository: its experiment register, its gates (pre-registered tests with IDs such as G1b or W1-A), its corrections, and the papers its experiments were built on. The numbers are real and so are the mistakes. Replace them with your own as you go.

**Structure.** Each chapter covers one step of the method or one body of knowledge. Within a chapter: the principle; the statistics or mechanism behind it, derived rather than asserted; the alternatives and why they were not chosen; the places in the repository where it was applied, with the actual numbers; the places where it was violated and what that cost; the likely objections, with answers; and exercises whose output is a committed file.

**On a phone or e-reader.** The book is built as an EPUB after every change and kept as one download: the
`book-latest` release of the repository (GitHub → Releases → *Book (latest build)* → `road-to-robotics-book.epub`).
Rebuild it with `bash journal/book/epub/build_epub.sh` (pandoc ≥ 3.0).

**Conventions.** Numbers cited are from `journal/EXPERIMENTS.md` unless a paper is named. Gate IDs are the register's. "You" is the person running the lab. "The tools" means the coding agent on the edge computer and the assistant in chat, whose role is discussed in chapter 15.

**Updated 2026-10-30.** The register grew from 15 to 21 tested papers and the lab's thesis changed from "what a JEPA world model buys on cheap hardware" to "what survives when a paper leaves the lab" (chapter 15.4a). New: chapter 11b; sections 2.6a (claims as rows, computed contradictions), 4.5a (verdict maps), 5.6a (withdrawn and void), 6.4a (the red-channel extract), 6.7a (a check that voids the run), 9.3a (the claims layer), 12b.8b–c, 12c.5a, 13a.10a–b, 11a.8a, 15.4a; the interview opener (16.2) rewritten. The book's method did not change; its scope did.

## Contents

### Part 0 — Prerequisites (start here if deep learning or PyTorch is new to you)

- **F.** The fast track: learning this with little time (produce, do not re-read; a 45-minute daily session and a weekly mock; flashcards; what to cut; a four-week plan)
- **P1.** Deep learning from zero: loss, gradients, backpropagation, layers, generalisation, optimisers, attention, JEPA and collapse, CEM, policies and flow matching, evaluation statistics, hardware; worked numbers; vocabulary table
- **P2.** PyTorch for the coding round: tensors and shapes, numerics, autograd, modules, layers, losses, optimisers, data, the training loop, reproducibility, mixed precision, checkpoints, DDP, debugging, the primitives for every item of the coding list, and two worked builds (an autograd engine from scratch, a tiny GPT end to end); all code run

### Part I — The method

1. Start from a decision
2. Translate the claim
- 2a. How to read a paper, and how to reason from it to an experiment
3. The baseline is the method
4. Pre-registration
5. One factor at a time, and the three verdicts
6. Trust the instrument before the result
7. Uncertainty: episodes, pairs, small n
8. Reading a failure
9. The register

### Part II — The knowledge, on your hardware

10. The hardware, from first principles
- 10a. Robotics fundamentals for learning engineers: frames and transforms, forward and inverse kinematics (with the SO-101 URDF), actuators, PD control, gains and delay, the two clocks, action spaces, cameras and calibration, simulation, safety; all code run
- 10b. The data pipeline: the LeRobot v3 format on disk, recording and its checklist, loading windows and padding, splits and normalisation, the pre-training audit, rollouts as data, provenance
11. Policies, chunks and boundaries
- 11a. The VLA policy papers, in depth (SmolVLA, TurboVLA, VLA-JEPA, REKI, RTC, VLAs-as-Tools, augmentation in post-training, SO-101 benchmark, FineART / G0.5 System 2 — running)
- 11b. Chunk length, candidate selection and denoising depth: inference-time papers and the tick budget (AAC, PACT-VLA, Horizon × depth, VeriVLA, FlashRT-PAI)
12. JEPA world models: what they are and what they are for
- 12a. The JEPA world-model papers, in depth (DINO-WM, LeWM, V-JEPA 2.1, RoboJEPA, H-JEPA, TD-JEPA, DeepJEPA, ContactGuard)
- 12b. Planning with JEPA world models, and grounding them (JEPA-WM study, Planning Limits, D-JEPA, PSG-JEPA, SA+IDM, CF-JEPA, Imagine-RL, FOREWARN, LaLa-World, HWM, SURVEYOR)
- 12c. The foundations, and why the loss is not the result (LeJEPA, V-JEPA 2, one-step fidelity, decision-relevant error, SALT, LeJEPA identifiability on the logged joints)
- 12d. Video world models as policies, and how they differ from JEPA (generative vs joint-embedding prediction, world action models, five ways a video model becomes a policy: Cosmos Policy, LingBot-VA, VPP2, Fast-WAM, VERA, GreenWAM; the side-by-side; what the evidence supports; what it means on a Jetson)
13. Judges, monitors and data
- 13a. The judge and monitor papers, in depth (FAIL-Detect, SAFE, ReAP, FARM, Robometer, STDR, FailSafe, VLA-Corrector, Spotter, No Free Checker, PRIME — void, RoboFQE)
- 13b. Which demonstrations help: the data papers, and the judge that chooses them (AXIS-Bench, CUPID, QoQ, Ego4WAM, coverage shift)
14. Engines, the harness, and the training basics
- 14a. Harness, deployment-time learning and perception papers, in depth (RegenHarness, HexaShell, SEAR, Robo-COP, Mulligan, ReTrace, COAP, HyMeS, residual RL ×2, SEVO, PEEK, blind assembly)
- 14b. LeRobot from the inside: adding a policy (config, policy, processors), the plugin system with a tested minimal plugin, serving policies remotely (LeRobot remote inference) and proprietary policies through a hosted API (Pi API beta)

### Part III — Practice

15. How decisions were made, and how they should be
16. The interview: the loop shape from public reports, what a frontier-lab founder says he hires for (attention to detail across the stack, signalling instead of credentials), the project deep-dive, paper discussion, live ML coding and the broken-script round, fundamentals, embodied-reasoning questions, take-home, behavioural
- **Appendix S.** Worked solutions to the exercises of P1, P2, 3, 7, 8 and 10a (and the book errors they found)
- **Appendix R.** The reading list: every paper the book relies on, in four tiers, with depth, hours, the chapter that summarises it and whether it is in the knowledge base
- Exercises: `exercises/coding/README.md`, the live-coding list (24 items, with a priority order and a four-week schedule)

## Reading order

- **Little time (most readers):** chapter F first; it says which parts of everything below to do, skim and skip.
- **New to deep learning:** P1 on paper with its exercises, then P2 with a Python session open, then Part I in order.
- **Already fluent in PyTorch:** Part I in order, then Part II, then Part III.
- **The "a" chapters** (2a, 11a, 11b, 12a–12c, 13a, 13b, 14a) are the papers behind the experiments, read from their method and experiment sections: objective, protocol with n and seeds, numbers with table references, what was not run, and what each means on this hardware. Each ends with a side-by-side table of how that family of papers evaluates and a list of corrections to the register. Read each after its base chapter, one family per sitting. Chapter 2a is the general method for reading any of them.
- **In a hurry before an interview:** 1, 3, 4, 7, 15, 16 (start with 16.1c and 16.7b), then 12; and the coding list, items 14, 23, 02, 03; the rehearsal files in `exercises/interview/`.

The reading list with depths is Appendix R (`91-reading-list.md`); the four-week exercise plan is Part V of the same file; the chapter exercises here are the longer form of those.
