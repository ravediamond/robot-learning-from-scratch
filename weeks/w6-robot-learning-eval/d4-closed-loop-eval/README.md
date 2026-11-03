# W6 D4: A closed-loop evaluator, and why val loss lies (capstone, part 1)

Book: chapters 3 (baseline), 5 (one factor, three verdicts), 6 (trust the instrument), 7.
From today, every policy you build is judged by THIS evaluator, not by its loss.

## What a real evaluator does
1. **Fixed start states**, generated once from a seed and reused for every policy: comparisons
   are PAIRED (same starts), the variance of start difficulty cancels.
2. **Per-episode outcomes** (success, collision, steps), not just a rate: you need them for
   bootstrap, McNemar, and to read failures.
3. **Same execution settings** for everyone (H_exec, T_MAX, clipping).
4. **Reports with intervals** (Wilson) and the paired table:
   `n11` both succeed · `n10` A only · `n01` B only · `n00` neither.

## The experiment
Train the W4 MSE head and flow head (`policies_ref.py`) on the same bimodal demos.
Measure (a) val chunk MSE on held-out demo episodes and (b) closed-loop success on 200 fixed starts.
Write down which ranks first on each BEFORE you run.

## Today's build (`w6d4_eval.py`)
`fixed_starts`, `evaluate(act_fn, starts, H_exec, seed)`, `paired_counts`, `val_chunk_mse`.

Run: `uv run pytest weeks/w6-robot-learning-eval/d4-closed-loop-eval -s`
