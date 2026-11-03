# The opener, with the 16.1c clause (16.2), and its three-level ladder (16.2a)

Spoken at ~150 words a minute. The `[30 s]` marks are where you should be; if you are behind at a mark, drop the next sentence in brackets.

> I re-run published robot-learning claims on an SO-101 and a Jetson Orin NX — the hardware people can buy — with the cheapest baseline measured first and the pass mark committed to git before any number exists. Twenty-one papers so far, about $45 of rented compute, every result and every correction public. `[30 s]` Four reproduce, twelve fail, three are not proven, one I withdrew. The fails are rarely "the method does not work"; they are "the cheap thing did as well" or "the budget is not there". The commanded joints rank action chunks better than a JEPA world model — 0.68 to 0.90 against 0.60 — yet the same model reads the task outcome at rest perfectly, AUROC 1.0: here it is a verifier, not a planner. `[60 s]` Episode length beats a 4B reward model as a failure detector, 0.94 against 0.82. PACT-VLA's chunk selection replicates offline — 0.27 degrees closer to the demonstration than a random pick, interval clear of zero — but four candidates cost 37 control ticks on the Orin instead of 8, so the paper's "no added delay" fails. (A gradient-free prefix overwrite I wrote cuts the chunk-boundary jump 71 to 73 percent for 0.4 milliseconds in vla.cpp.) `[90 s]` The corrections I am proudest of came from going down the stack: a `strict=False` load that left an encoder at random init, a register row whose unescaped pipes faked a fail, and a data loader that fed two models the red channel only — each caught by a check, each corrected with the old row kept. On the arm: REKI, trained on my data, put the bowls in 19 of 20 and 19 of 20 times.

Word count: about 280 (≈ 115 s at 150 wpm; the bracketed sentence is the one to drop if you are behind).

Rewritten 2026-10-30 for the new thesis (chapter 15.4a). Changed, with the row that decided each: "$32 across fifteen papers" → "about $45 across twenty-one" (register cost fields, $45 as summed for the gate-protocol post of 2026-10-29; every row since is $0); the JEPA-question framing → the three conditions and the verdict counts (claims build, 2026-10-30); the Robometer-vs-episode-length and PACT-VLA results added; the 27B-judge clause and "two voided gates" removed for time (both stay in the ladder); the red-channel correction added (WS-047 correction row, 2026-10-28). Previous version: git history of this file.

## The ladder: three levels for every number

Level 1 = the number and its n. Level 2 = the mechanism (chapter 8's form). Level 3 = the alternative you did not take, and why.

**About $45 of compute.** L1: $45 summed from the register's cost fields for the gate-protocol post (2026-10-29), retries included; the vast invoices including retries came to $46.3–47.3; every row since is $0; 21 papers. (Was $31.65 over 15 papers on 2026-10-27.) L2: most gates run on the Orin for free; rented GPUs only for training and serving models the Orin cannot run. L3: a desktop GPU would remove the rental friction; not bought because the thesis is what the low-cost rig can show (rig page, "not owned yet").

**G0: 0.31 vs 0.89.** L1: open-loop error at K = 30, world model 0.31 vs copy-last 0.89; every corruption ratio ≥ 1.3; 2026-10-11, one seed. L2: the latent tracks the arm because the arm is most of what moves in the frame. L3: a second seed and a held-out scene; not run because G1b decided the planning question first.

**G10: AUROC 1.0.** L1: W1 outcome readout at the judge's time, AUROC 1.000 (W2 0.97); the colour judge as teacher; 2026-10-17. L2: at rest the bowls' configuration is a static, visible fact. L3: the shadow veto (predicting failure 30 steps ahead, +0.179 over policy identity offline) — not deployed until a fresh-rollout gate against joints only.

**G1b: 0.60 vs 0.68–0.90.** L1: top-1 among 8 chunks, 0.60 for the world model vs 0.68–0.90 for an action-only baseline; 25 val episodes, 703 anchors. L2: position control — the decoys differ in arm state, so the commanded joints already separate them; pairwise 0.92 compounds to ~0.56 over 7 comparisons. L3: decoys that share the arm state (object-only differences); that is the object-state test.

**Live thresholds flagged every success.** L1: W1-A, the 2.2 / 10.0 reflex flagged 42/42 successes on the 50 judge episodes (2026-10-19); refit (G11-B-cal) catches 3/25 failures at 1/44. L2: thresholds fit on seen rollouts; the fresh distribution is wider. L3: a conformal band — tried as W1-C2, not conformal across time when μ, σ come from the same successes.

**Judge with zero false positives.** L1: G9c-3, deterministic colour judge at rest, FP 0/25 [0, 13.3 %], FN 5/62, 94.3 %. L2: masks and geometry on the frame after the arm stops; no learned part to drift. L3: Robometer-4B as a zero-shot judge — failed (G9a-AB complete, pooled 0.818 vs bar 0.85; episode length alone 0.938).

**27B VLM usable on the whole episode.** L1: G-flora-abl, Qwen3.5-27B on the 64-frame side-by-side clip: 2/42 successes flagged, 7/8 failures, AUROC 0.91 (2026-10-20). L2: the input format and prompt did the work, not Flora's robot fine-tune; 9B failed (16/42). L3: sliding windows with Gemini — the judge-τ fit caught 0/8 (2026-10-19).

**Prefix overwrite: 71–73 % for 0.4 ms.** L1: jump ratio cut 71–73 % against naive async at d = 8, 12, 20; 254 boundaries per delay; +0.4 ms per chunk. L2: each chunk starts from fresh noise; overwriting its prefix with the queued actions ties it to what the arm is doing. L3: RTC's guided inpainting — not available in vla.cpp because ggml lacks backward rules for flash attention and concat.

**Two voided gates.** L1: G11-A (random encoder; 192 missing / 192 unexpected keys) and G12 (135.4 + 230.3 ≠ 318.2 ms), both 2026-10-18. L2: `strict=False` + a key remap; a stand-in model in the benchmark. L3: catching them before publishing — the strict-load check and the parts-must-sum rule are now run before any number is reported.

**Two dollars, 0.020°.** L1: path A, the authors' trainer, full LIBERO init, 35k steps: task swap 0.020°, per-layout 38.9 %; ≈ $2.0 (+ ~$0.8 failed starts); control 0.105 normalised ≈ 5.5° of pan. L2: the instruction is redundant on ~94 % of the frames; the pose leaks the side. L3: data that forces the instruction (layouts where the pose does not decide) — the fix is in the data, not the model.

**REKI 19/20 and 19/20.** L1: two instructions, 20 trials each, 25 s, judged at rest (2026-10-27); SmolVLA was 9/10 under the same protocol. L2: register-token memory — oscillation 1/40 vs 20 % on earlier policies. L3: the K = 1 (memory off) ablation on the same batch; not run yet.

**4 / 12 / 3 / 1 out of 21.** L1: paper verdicts from the claims build, 2026-10-30: reproduces SmolVLA, VLA-JEPA, REKI, LeJEPA identifiability; fail LeWM, DINO-WM, RoboJEPA, H-JEPA, ContactGuard, FAIL-Detect/SAFE, ReAP, VLA-Corrector, Robometer, SURVEYOR, AAC, PACT-VLA; not proven RTC, HWM, Horizon × depth; withdrawn TurboVLA; D-JEPA not yet tested. L2: a paper's verdict is computed from its own stated claims that were tested, never from rows written by this rig. L3: per-claim verdicts instead of per-paper; that is what the claims table on the site shows, and the per-paper word is a summary of it.

**Episode length 0.94 vs reward model 0.82.** L1: G9a-AB, failure AUROC, episode length 0.938 vs Robometer-4B pooled 0.818 (bar 0.85). L2: on this task failures run long (stalls, retries); a 4B model trained on other robots does not know this task's timing. L3: a LoRA-adapted Robometer (G9b); never run past preprocessing.

**PACT-VLA: −0.27°, 37 ticks vs 8.** L1: 20 episodes, 254 boundaries, d = 8; PACT − random −0.27° (CI −0.44 to −0.09), PACT − stale −0.27° (−0.48 to −0.09); cycle 1,200 ms vs 237.5 ms (2026-10-29/30). L2: the fresher frame does the work (stale selection = random); vla.cpp generates the 4 candidates one after another and vision is 151 ms of each. L3: a batched sampler in vla.cpp; not written, because even 4 batched candidates plus the pre-commit vision pass would need to fit in one chunk's time.

**The red channel.** L1: WS-047 extract read `it[cam][0]` on a one-timestamp item, i.e. the red plane; found 2026-10-28; LeJEPA mean R² 0.624 → 0.565, verdict unchanged; the first HWM run declared invalid before scoring. L2: the self-check passed because it asked for four timestamps, a different tensor shape. L3: shape assertions at every positional index, and the self-check run with the experiment's own arguments.
