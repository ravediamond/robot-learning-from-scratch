# W8 D4: Re-derive your own published numbers

Book: 9 (the register), 15 (how decisions were made), 16 (the interview). Your files, not toys:
- `reference/journal/papers/failure-monitors/results/counts_ci.json`: k/n counts with the Wilson
  intervals your failure-monitors paper reports (judge τ, G16-B, W1-A surprise, G-flora, ...).
- `reference/journal/papers/jepa-world-model/gaps/anchor_predictions.csv` + `anchor_ci.json`:
  per-episode monitor scores (W1, W2p, joints features × 5 seeds) on your labelled bowls rollouts,
  and the AUROCs + CIs you published from them.

**Rule for the interview: a number you cannot re-derive from raw files with your own code does not
get said out loud.** Today you earn the right to cite these.

## What "re-derive" means
1. Parse the raw file yourself.
2. Recompute with code you wrote this course (no copy from the original scripts).
3. Compare with the stored number at a stated tolerance. Explain every difference (rounding, the exact
   z quantile, a different bootstrap RNG) or flag it as an error.

## Today's build (`w8d4_rederive.py`)
`wilson`, `audit_counts`, `load_anchor`, `auroc`, `recompute_anchor`, `bootstrap_auroc_ci`.
Then write in PROGRESS.md: the three numbers you would now quote, with their intervals, in one line each.

Run: `uv run pytest weeks/w8-robotics-register/d4-rederive-your-numbers -s`
