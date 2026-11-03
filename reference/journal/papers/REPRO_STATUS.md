# REPRO_STATUS — paper repos × reproduction class (WS-117, 2026-11-03)

**Rule:** a paper number is reproducible only if the repo stores the rows it is computed from (per episode, per seed, per boundary) and `make check` recomputes it on CPU from those rows; a number that needs an image, a GPU or a download is L1 with its command, never L0.

| repo | numbers | L0-rows | L0-summary | L1 | unsourced | `make l0 paper` (2026-11-03) | audit commit |
|---|---|---|---|---|---|---|---|
| paper-reward-models-stopping-rule | 72 | 64 | 8 | 0 | 0 | pass | e2e8956 |
| paper-vla-edge-inference | 52 | 17 | 30 | 0 | 5 | pass | 5e88f60 |
| paper-register-reproducibility | 84 | 77 | 7 | 0 | 0 | pass | 61fbc4e |
| paper-jepa-world-model | 135 | 57 | 77 | 1 | 0 | pass | c596ef9 |

Per-repo tables and the reason each residual group stays L0-summary: `REPRO_AUDIT.md` at each repo's root. C's 5 'unsourced' are protocol-commit hashes checked against git history, not files. No number or verdict changed; no file moved to the Hub (none over 50 MB). Not done (not checked this pass): folding in post-cut results the drafts cite as pending (C: WS-064 gate A rows, WS-086 RACE, WS-089; JEPA anchor predictions are already stored as `anchor_predictions.csv`). Known gap: `~/robot-data/robometer/` is absent on the Orin, so B's Robometer per-episode scores are still missing. The TOPReward rows are already stored (`repro/topreward/expected/{ab,own}.jsonl`).
