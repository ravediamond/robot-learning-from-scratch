# Chapter 6. Trust the instrument before the result

## 6.1 The principle

Before you believe a number, verify that the apparatus that produced it measured what you think it measured. A measurement bug produces a plausible number with no visible symptom, and nothing downstream can detect it: passes feel earned, fails feel informative, and both are noise.

Four checks run before every gate in this repository. Each exists because of a specific failure.

## 6.2 Strict checkpoint loading

**The failure (G11-A, 2026-10-18).** The W1 surprise benchmark on the Orin reported 12.78 ms per dual-camera check, under the 20 ms bar: pass. The next day, the still-scene noise-floor script tried `load_state_dict(strict=True)` on the same checkpoint inside the same Docker image and found 192 missing and 192 unexpected encoder keys: the image shipped transformers 5.5.4, the checkpoint was saved under 5.19.0, and the key names differed. The G11-A benchmark had used `strict=False` plus a hand-written key remap that did not match either version. Result: the ViT encoder was randomly initialised; every latent value in G11-A was meaningless; only the timing stood (same architecture, same FLOPs).

**The check.** `scripts/check_strict_load.py`: load with `strict=True`; print and count missing and unexpected keys; refuse to proceed on any. Run it inside the exact environment the gate will use, not on the Mac. It is step 0 of `ROBOT_DAY_CHECKLIST.md` and runs before every Jetson gate in `GATES_RUNBOOK.md`.

**Why `strict=False` is dangerous.** It exists for legitimate partial loading (a backbone into a new head). Used on a full checkpoint, it silently leaves every unmatched tensor at its random init and returns normally. The model runs, produces outputs of the right shape with plausible statistics, and nothing warns you. The only defence is to never use it on a checkpoint you expect to be complete, and to assert the key sets match.

**Interview version.** "I voided a passing gate because the encoder was random. The timing was right and the latents were garbage, and no number in the output could have told me. Now strict load is step zero."

## 6.3 Parts must sum

**The failure (G12, 2026-10-18).** The REKI 2.4B feasibility benchmark on the Orin reported 318.2 ms per chunk against a 250 ms bar: fail; decision: drop REKI, port register tokens into SmolVLA. The review (`V1.0_REVIEW.md` §4) found the script timed a random-init Qwen2.5-1.5B with no vision tower and no Gated DeltaNet layers, plus a hand-written random expert. The stated parts (135.4 ms text, 230.3 ms expert) do not sum to the stated total (318.2 ms). The benchmark measured a stand-in, and the decision was taken on a number that measured nothing.

**The check.** Time each component and the whole, in the same run, and assert they agree within measurement noise. If they do not, something is not running, is running twice, or is being timed in the wrong place. The REAP latency table (W1 check 17.6 ms; imagination 153–158 ms; total ~224 ms) was built this way: components placed next to the measured total so the reader can add them.

**The general form.** Any pipeline with stages has conservation laws: frames in = frames out; episodes in the split = episodes in the report; sum of per-task counts = total; parameters loaded = parameters in the checkpoint. Assert them. The G9b re-analysis found the first summary had counted 400 teleop demos as successes because the episode counts did not reconcile with the protocol's test set.

**The same list, seen as hiring evidence.** 6.2 and 6.3 are the first two rows of 16.1c's table: a fault that presented as a model result (G11-A) or a model's latency (G12) and lived in the loader and in the benchmark. Read from the other side of an interview, the check that caught each is the evidence of attention to detail below the model; 16.1c lists the rest, and the public ledger (`/faults/`, WS-044) sorts all of them by the layer they lived at.

## 6.4 Parity with the reference under shared randomness

**The need.** Every converted or ported model (a GGUF export, a vla.cpp port, a checkpoint moved between library versions) can silently differ from the reference. The TurboVLA port differed in architecture; the GGUF SmolVLA could have differed in weights, precision or preprocessing.

**The check.** Run the reference (PyTorch) and the port on identical inputs and compare outputs. For stochastic heads (flow matching samples noise), the noise must be shared: export the exact noise tensor from one side and feed it to the other, or the comparison measures sampling variance, not implementation difference. The GGUF check: 20 fixed frames, shared noise, median max |Δ action| 0.80° against a 1° line; one outlier frame at 11.4°; and, for context, PyTorch's own noise-to-noise spread of 12.2° so the reader knows what "different" means on this head. The f32-matmul caveat is recorded.

**The line.** 1° was chosen as "below what the servo resolves"; it is a judgment, written down before the comparison. A parity check without a pre-stated tolerance is a number you will rationalise.

**When no reference exists.** State that parity could not be checked and treat the result as provisional. The LIBERO-Goal task-swap control for TurboVLA path A (released checkpoint, upstream code, 0.105 vs 0.095) is a parity check of a different kind: it establishes what "the instruction is used" looks like for a known-good model, so the port's 0.1–0.25° and path A's result can be read against it.

## 6.4a One channel instead of three

**The failure (WS-047, found 2026-10-28).** The LeJEPA identifiability extract read each frame as `it[cam][0]`. With several timestamps a LeRobotDataset item for a camera is `(T, 3, H, W)` and `[0]` is the first frame; with one timestamp it is `(3, H, W)` and `[0]` is the **red plane**, which normalisation then broadcast to three channels. Both encoders saw red-only images. Every downstream number was plausible; B1–B3 were met. The bug was found while checking the HWM run (WS-051) against G0, because the probe's own self-check asks for four timestamps and therefore passes.

**What it cost and what it did not.** The LeJEPA verdict held; its numbers changed (mean R² 0.624 → 0.565, anisotropy 0.582 → 0.639) and both rows stay in the register. The first HWM run used the same extract and was declared invalid before scoring.

**The check.** Any loader that indexes a tensor by position asserts the shape it expects (`assert x.ndim == 4`), and the self-check runs with the exact arguments the experiment uses, not convenient ones. And the cheapest detector of this class: **reproduce a known number on the new extract** (6.7) — the corrected HWM latents had to give G0's copy-last error (0.915 vs 0.899) before scoring.

## 6.5 The smoke run whose output should be meaningless

**The idea.** Run the full gate pipeline on a model you know is useless (a 4-step checkpoint, or random weights). If the output looks reasonable, the pipeline is not measuring the model. If the output looks like what random weights should produce (collapse checks failing, AUROC near 0.5, loss near log(number of candidates)), the pipeline is probably measuring what you think.

**In the repo.** The path A sensitivity gate was run on a 4-step smoke checkpoint before the real training ("results are meaningless and stay in outputs/"). The D-JEPA training smoke on the Orin reported the decision term "starting near log 8 minus masked candidates", which is what an untrained ranker over 8 candidates should show. The collapse checks failed on the 8-step run, as expected. Each of these is a confirmation that the instrument responds correctly to a known-bad input.

**The complement.** Also run it on a known-good input when one exists: the G7a scorer was run with W1 standing in for W2 on the 69 published rollouts, and the arm-pose probe came back at R² 0.56 against the recorded 0.57. That reproduction of a known number is the cheapest possible instrument calibration.

## 6.6 Environment pinning

The G11-A failure was a library-version mismatch. The defence is to record and pin: the Docker image tag, the transformers version that saved the checkpoint, the lerobot commit, the vla.cpp commit and ggml tag, the GGUF sha256, the dataset commit (the 69 rollouts are pinned at `bad1c8e`). The register's source links point at files that record these. Path A's remote job needed a two-environment design because LeRobot and upstream's transformers 4.56 want incompatible `huggingface_hub` versions; that incompatibility would have been a silent wrong-version load without the fail-fast import check added after the first failed starts.

## 6.7 Reproducing a known number first

Before measuring anything new on a changed setup, reproduce one old number. The D-JEPA protocol's step 0 reproduced W1's stored G1b regret exactly on the Orin before training the operator. The G7a dry run reproduced the arm-pose R². The GGUF parity reproduced PyTorch actions. Each took minutes and would have caught an environment, data or code drift before it contaminated a new result.

## 6.7a A check that voids the run before any number is read

Strict load, parts-sum and parity check the code. Some methods also need a check that the *instrument the method builds* works on this data before it is pointed at the question. PRIME (13a.10a) builds a progress graph from teleop; the protocol required that held-out teleop episodes get monotone progress (Spearman ≥ 0.8) before any rollout was scored. It got 0.51, and the run was declared void. The script had already written the rollout scores; they were not read.

Two disciplines make this work. The check and its threshold are in the frozen protocol, so a failed check cannot be renegotiated after seeing the rollout result. And "not read" is literal: the scores are not looked at, not reported, not summarised as "would have been". Honouring a void is the cheapest proof an interviewer can be given that the bars are real.

## 6.8 What the interviewer will ask

*"How do you make sure a benchmark number is real?"* The four checks, each with its failure. "Two of my gates were voided in one day; now these run first."

*"What's the most embarrassing bug you've found in your own evaluation?"* G12 or G11-A, told plainly, with the fix. Interviewers at a company that scaled Autopilot and Optimus have seen worse; what they are checking is whether you found it yourself and changed the process.

*"How do you check a model port?"* Shared-randomness parity with a pre-stated tolerance, plus a known-good control.

## 6.9 Exercises

1. Write `check_parts_sum.py`: given a timing JSON with component and total fields, assert agreement within a tolerance and print the discrepancy. Run it on the REAP latency table and on the voided G12 numbers.
2. Reproduce the GGUF parity check from its script with a different seed. Does the 11.4° outlier move? Write two sentences on what that tells you about the tolerance.
3. List every version, commit and hash a reader would need to reproduce G13b exactly. Check which of them are actually recorded in the repo. File the missing ones.
