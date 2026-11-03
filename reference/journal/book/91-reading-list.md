# Appendix R. The reading list

Every paper this book relies on, in the order to read them, with how deeply, how long it takes, which chapter already summarises it, and whether the PDF is in the project knowledge base (KB). It supersedes Appendix B of `journal/METHOD.md`, which predates chapters 12c, 12d, 13b and 14b.

**Depths.** **D4, derive:** every section; reproduce the key equation or algorithm on paper; explain the objective and the main table from memory; 3–4 hours. **D3, implement-level:** method, experiments and appendix; know the hyper-parameters, the ablations, the baselines run and not run; 1.5–2 hours. **D2, claim and table:** abstract, method figure, main table, limitations; 30–45 minutes. **D1, one line:** abstract and figure; 10 minutes.

**How to read** any of them: chapter 2a (reading order, the tells, the template). The book's chapter for each paper is your answer key: read the paper first, fill the 2a.8 template, then compare with the chapter.

**If time is short** (chapter F): Tier 1 only, then the Tier 2 papers closest to the role you are interviewing for. Tier 3 is read the week its experiment runs; Tier 4 when a question touches it.

## Tier 1 — the core (≈ 25 hours; read in this order)

| # | Paper | arXiv | Depth | Chapter | KB | Why it is here |
|---|---|---|---|---|---|---|
| 1 | LeWorldModel (LeWM) | 2603.19312 | D4 | 12a | yes | The world-model recipe used throughout; derive SIGReg; know the table the G0 bar came from |
| 2 | LeJEPA | 2511.08544 | D3 | 12c | yes | Why SIGReg works: the isotropic-Gaussian argument and the Epps–Pulley test |
| 3 | V-JEPA 2 (incl. V-JEPA 2-AC) | 2506.09985 | D3 | 12c | yes | The reference JEPA and the first action-conditioned robot planner built on it |
| 4 | SmolVLA | 2506.01844 | D4 | 11a | yes | The policy run most; flow-matching expert, chunk size, its ablations |
| 5 | Real-Time Chunking (RTC) | 2506.07339 | D3 | 11a, 11.4 | yes | The chunk-boundary problem and its reference solution |
| 6 | JEPA world models for planning (Terver et al.) | 2512.24497 | D3 | 12b | yes | The design-choice study whose protocol everyone inherits |
| 7 | Planning Limits of Latent World Models | 2609.39235 | D3 | 12b | yes | Why planning failed (G1b): range = rollout length, even with a perfect simulator |
| 8 | One-step fidelity is a poor proxy for planning | 2610.19375 | D3 | 12c | yes | Loss is not the result (r ≈ −0.01); the matched-control lesson |
| 9 | LaLa-World | 2610.27851 | D3 | 12b, 12d | yes | The newest LeWM-line model; the one matched latent-vs-generative comparison; the action-sensitivity test (§28) |
| 10 | DINO-WM | 2411.04983 | D2 | 12a | yes | The baseline every latent-planning paper compares against |
| 11 | TurboVLA | 2607.27205 | D3, then §4–5.1 at D4 against the code | 11a, 8 | no | Your own correction story; know the divergences and Table 3 |
| 12 | VLA-JEPA | 2602.10098 | D3 | 11a | yes | World-model loss used at training time only; the upstream leak fixed |

## Tier 2 — read before an interview (≈ 25 hours)

| # | Paper | arXiv | Depth | Chapter | KB | Why |
|---|---|---|---|---|---|---|
| 13 | SO-101 VLA benchmark | 2606.08881 | D3 | 11a | yes | The only external reference on your arm; the wrist-camera and failure-taxonomy claims |
| 14 | ContactGuard (LeWM pre-contact monitor) | 2608.13438 | D3 | 12a | yes | The paper W1-C1 tested and failed to transfer |
| 15 | D-JEPA | 2609.24749 | D3 (§3 slowly) | 12b | yes | Loss term vs a set operator trained on executed outcomes |
| 16 | SALT | 2609.33595 | D3 | 12c | yes | Worse one-step error, better planning; the mismatch-failure ratio |
| 17 | Decision-relevant prediction error (DRPE) | 2609.32322 | D2 | 12c | yes | 1 % total-error gap → 60-point planning gap |
| 18 | V-JEPA 2.1 | 2603.14482 | D2 | 12a | yes | The dense-loss update to the reference encoder |
| 19 | REKI | 2610.22178 | D3 | 11a | yes | Memory in the action expert; the oscillation criterion reused in G14 |
| 20 | FAIL-Detect | 2503.08558 | D3 | 13a | yes | Conformal failure detection; why the W1 band was not conformal |
| 21 | SAFE | 2506.09937 | D2 | 13a | yes | Failure probe on a VLA's own features |
| 22 | ReAP | 2610.16309 | D3 | 13a | yes | The risk head behind REAP v1/v2 |
| 23 | Robometer | 2603.02115 | D2 | 13a | yes | The reward model episode length beat |
| 24 | No Free Checker (verifier survey) | 2609.09250 | D3 (§5) | 13a | no | The taxonomy behind the verifier tiers |
| 25 | CUPID | 2506.19121 | D3 | 13b | yes | Which demonstrations help, measured on closed-loop return |
| 26 | Coverage shift in VLM success monitoring | 2610.24801 | D2 | 13b | yes | Better policy → harder failures → a fixed judge degrades |
| 27 | Fast-WAM | 2603.16666 | D2 | 12d | no | Video world models do not need test-time imagination |
| 28 | Cosmos Policy | 2601.16163 | D2 | 12d | no | A video model turned into policy, world model and value function |

## Tier 3 — read the week the matching experiment runs (D2, then D3 before its bar is final)

| Paper | arXiv | Chapter | KB | Experiment it feeds |
|---|---|---|---|---|
| Adaptive Action Chunking (AAC) | 2604.04161 | 11b | no | WS-049, done (fail) |
| PACT-VLA (pre-commit candidate selection) | 2610.28019 | 11b | no | WS-059, done (offline pass, Orin fail) |
| Horizon × depth (Euler steps vs commitment) | 2610.32102 | 11b | no | WS-057, done (not proven); arm gate B pending |
| VeriVLA | 2610.21985 | 11b | no | §2 fourth monitor (two-sample trigger) at the next rollout session |
| FlashRT-PAI | 2610.24349 | 11b | no | nsys trace of vla.cpp, bar frozen |
| FineART + LeRobot "Prompting Robots" Part 2 (G0.5) | 2609.36416 | 11a | no | WS-062, running |
| HWM (hierarchical latent world models) | 2604.03208 | 12b | no | WS-051, done (not proven) |
| SURVEYOR | workshop poster | 12b | no | WS-050, done (fail) |
| When Does LeJEPA Learn a World Model? | 2605.26379 | 12c | no | WS-047, gate 0 done (reproduces); gate 1 needs the arm |
| PRIME (experience-graph progress) | 2610.23769 | 13a | no | run 2026-10-29, void; a new protocol must fix thresholds first |
| RoboFQE | 2610.18699 | 13a | no | NEXT_ANGLES §11 evaluation day (reported, not gated) |
| PSG-JEPA (physical-state grounding) | 2608.06799 | 12b | yes | §7 three-arm grounding test |
| SA+IDM / Genisom (state alignment) | 2609.03565 | 12b | yes | grounding variants |
| CF-JEPA | 2610.00727 | 12b | yes | §8 participation ratio, distractor robustness |
| Quality over Quantity (QoQ) | 2603.09056 | 13b | yes | §24 curation scores |
| Quality-aware VLA pretraining (AXIS-Bench) | 2610.18869 | 13b | yes | §24 weighting vs filtering |
| Ego4WAM | 2609.40341 | 13b | yes | §25 egocentric video into W1 |
| RoboJEPA | 2610.10515 | 12a | yes | the scaling angle |
| H-JEPA | 2610.06805 | 12a | yes | G6b (hierarchical planning) |
| Temporal-Distance JEPA | 2607.25337 | 12a | yes | cost functions for planning |
| FARM | 2609.11445 | 13a | yes | small readouts on frozen policy states |
| VLA-Corrector | 2609.06508 | 13a | yes | verifier + recovery prompt |
| Spotter | 2609.36808 | 13a | yes | parallel VLM screening, retreat arm (§4) |
| Mulligan | 2610.05882 | 14a | yes | §11 initial-state selection (Sobol) |
| ReTrace | 2610.15814 | 14a | yes | recovery data from own failures |
| HyMeS (memory as code, SO-101) | 2608.09410 | 14a | yes | memory for a Markovian VLA |
| SEVO (observation fixes on SO-101) | 2605.11114 | 14a | yes | camera placement, overlays (§21) |
| RegenHarness | 2609.27612 | 14a | yes | the harness contracts (v1.5) |
| Blind precise assembly | 2610.23849 | 14a | yes | proprioception-only control; Wilson bars as a reporting model |

The prep-queue papers (smoothness ranks skill, state criticality, RAFAIL, Robot-WATCHDOG, FailBench, the chunking-stability statistic, GeoAAC, fewer-steps, SMILE, RESTA, P² probes, GR-BC, VLA-Rolling, PICES) stay in `journal/PREP_QUEUE_2026-10-23.md` with their sections to read; each is read only when its run is scheduled.

## Tier 4 — skim when a question touches it (D1–D2)

| Paper | arXiv | Chapter | KB |
|---|---|---|---|
| SEAR (self-evolving agents benchmark) | 2610.18792 | 14a | yes |
| Robo-COP | 2610.09228 | 14a | yes |
| COAP / Embodied Turing Machines | 2610.12369 | 14a | yes |
| HexaShell | 2610.23479 | 14a | no |
| VLAs-as-Tools | 2605.13119 | 11a | no |
| Augmentation in VLA post-training | 2610.05994 | 11a | no (numbers kept in 11a.7) |
| DeepJEPA | 2610.00368 | 12a | no |
| Imagine-RL | 2609.24033 | 12b | no |
| FOREWARN | 2502.01828 | 12b | no |
| STDR | 2606.31377 | 13a | no |
| FailSafe | 2510.01642 | 13a | no |
| PEEK | 2509.18282 | 14a | no |
| Residual-foresight RL; HIR-ResRL | 2607.16506; 2606.22860 | 14a | no |
| Video Prediction Policy 2 | 2610.10270 | 12d | no |
| VERA (video models into policies) | 2605.27817 | 12d | no |
| GreenWAM | 2610.18774 | 12d | notes only |
| The Reality Gap in Robotics (survey; Annual Review 2026) | 2510.20808 | 10a.8, 12c.6a | no |

## Background outside this book's corpus (D2, for breadth in an interview)

Not summarised in any chapter; the papers an interviewer at a robot-learning company will assume you know by name and claim:

- **ACT / ALOHA** (action chunking with transformers): where chunking and the CVAE action head come from.
- **Diffusion Policy**: diffusion over action sequences; the multimodality argument of P1.14.
- **π0 and π0.5** (Physical Intelligence): the flow-matching VLA lineage SmolVLA follows, and the models behind the Pi API (14b.11).
- **OpenVLA**: the autoregressive-token VLA baseline.
- **UMI** (universal manipulation interface) and one egocentric-retargeting paper: the "human data without a robot" vocabulary.
- **LAPA or Genie** (latent actions from video without action labels).
- **DAgger**: the classic fix for compounding error, and its cost on a real robot (16.5a).
- **One sim-to-real paper** with domain randomisation (10a.8), for the "how would you go from sim to real" question.
- `TODO-verify` **"Neural thickets"** (Levine's one recommended paper of the last year, 16.1c; attributed on stage to Phillip Isola's group, first author not named — verify title and authors before citing): as models grow, random perturbations of the weights stop hurting, which reads as a subversive account of why RL works (noise, not credit assignment). Relevant to 14a's residual-RL answer; D1.

## Not papers, read at D3

- `journal/EXPERIMENTS.md`, `journal/LESSONS.md`, `journal/DECISIONS.md`: the register, once, cover to cover, with Part I open.
- The current architecture document (`latent-stack-architecture-v1.5.md`) and the data-engine use case (`journal/USE_CASE_KITTING_DATA_ENGINE.md`): the designs you would be asked to defend.
- LeRobot's `docs/source/bring_your_own_policies.mdx`, `remote_inference.mdx`, `modeling_smolvla.py` and the dataset v3 code: read as a maintainer would (chapters 10b, 14b).

**Totals at the stated depths:** Tier 1 about 25 hours, Tier 2 about 25, Tier 3 about 15 spread over the experiments, Tier 4 and background about 10. At four hours a week, Tiers 1–2 take about three months; at eight, six weeks. Chapter F's routine fits one Tier 1 paper per weekend alongside the coding practice.
