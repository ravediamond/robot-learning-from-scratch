# Self-improvement loop — pilot (the "7-day plan" of 2026-10-31, written down; decided 2026-10-31 evening: pilot only)

One iteration, manual reset, bowls task, the existing SmolVLA checkpoint as iteration 0. Decides whether the full
loop (FEASIBILITY.md §3, ≈ 14–18 arm days) is attempted.

- Iteration 0: `ravediamond/smolvla_2026-09-24_14-34-40` @ `c1d76f6`. 20 trials on a sealed held-out layout list
  (hash committed before trial 1, the §29 `LAYOUTS.csv` pattern), owner labels at rest.
- Batch: 40 rollouts on a second sealed list (training layouts, disjoint from held-out), judged by G9c-3 at rest,
  owner labels alongside (the verifier's errors are counted, never trained on: AGENTS.md §1 — failed rollouts
  never enter the training set).
- Fine-tune: iteration 0 data + the verified successes, same recipe as the smolvla post's compute block, vast
  (~$3), resumed from the checkpoint; seed and steps written before the run.
- Iteration 1: 20 trials on the same held-out list, same labels.
- Bar (frozen here): successes(it 1) ≥ successes(it 0) on the held-out 20. Below → full loop not attempted, the
  pilot is published as a negative (what the successes-only set did to the policy). At or above → build reset +
  watchdog (FEASIBILITY §3) and run K = 4.
- Cost: ~2 arm days, ~$3, one Orin evening for the judge pass.

## Amendment 2026-11-01 (WS-083, after reading ENPIRE, LoopWAM, HALTER in full — `RELATED_CARDS.md`)

The bar above is **not changed**. Three additions:

1. **Environment first (ENPIRE's order).** Before rollout 1, `env.py` is committed with three calls whose code is then
   frozen for the pilot: `clamp()` (joint / workspace limits and a stall timeout — the watchdog; a violation ends the
   rollout as a failure), `judge()` (G9c-3 with its unattended end-rule), `reset()` (manual in the pilot: the operator
   resets, the call only logs it). The commit hash is written in the pilot log; no rollout runs on uncommitted code.
2. **Arm 1b (LoopWAM-style).** Same 40 rollouts, same held-out 20,
   one more ~$3 fine-tune from the same checkpoint and seed: instruction suffix "Quality: q" (iteration-0 demos q*,
   verified successes q_s, failed rollouts q_min), action loss masked (λ = 0) on failed rollouts. At deployment the
   policy is conditioned on q*. Note from the read: SmolVLA has no video stream, so with λ = 0 a failed rollout adds
   **no** loss; 1b tests the quality token, not LoopWAM's use of failures. Reported beside 1a (the frozen
   successes-only recipe); the bar is read on 1a only (unchanged). Cost: + ~$3, + 0.5 arm day (20 trials).
3. **AGENTS.md §1, amended by the owner 2026-11-01:** "never imitate failed actions (action loss masked; observations
   may supervise)". Arm 1b is therefore within the rule, provided λ = 0 on failed rollouts is enforced in code and
   checked at fine-tune time (an assertion that the masked episodes contribute zero action loss, logged with the run).
   Arm 1a is unaffected. Order of the pilot: `env.py` frozen (watchdog first) → iteration 0 held-out 20 → 40 batch
   rollouts → fine-tunes 1a and 1b → held-out 20 each.

Reset for the full loop: cost the fixture against 2 taught reset skills (HALTER-style, `FEASIBILITY.md` §3) before
building either; the pilot keeps the manual reset.

Cost with the amendment: ~2.5 arm days, ~$6, one Orin evening for the judge pass, ~2 Orin days for `env.py`.
