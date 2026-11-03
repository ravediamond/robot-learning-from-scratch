# Robot Learning From Scratch

9 weeks. 1 hour a day, Monday to Friday. Goal: write PyTorch yourself, understand
every piece of a VLA + world-model stack, spot when generated code is wrong, and
pass the coding and research rounds of a top robotics startup.

No GPU needed. Everything runs on a Mac (CPU or MPS) in under 5 minutes.

## Setup (once)

```bash
uv sync                      # installs torch, numpy, pytest, matplotlib
uv run pytest weeks/w1-pytorch-basics/d1-tensors-shapes   # should FAIL: you have work to do
```

## A day (1 hour)

| Min | Step | File | What you do |
|---|---|---|---|
| 0–10 | Read | `README.md` | One page. The concept, nothing more. |
| 10–15 | Predict | `predict.md` → `predict_run.py` | Write your guesses in `predict.md` FIRST. Then run `uv run python predict_run.py` and compare. Every miss = something you did not understand. Write one line on why. |
| 15–45 | Build | `wXdY_*.py` | Fill the `TODO`s. Run `uv run pytest <lab dir> -k "not review"` until green. |
| 45–55 | Review | `review_*.py` | Code that looks right and is wrong. Find the bug, fix it, write a one-line explanation at the top of the file. `uv run pytest <lab dir> -k review` turns green when fixed. |
| 55–60 | Explain | `questions.md` | Answer out loud, as in an interview. 1–2 minutes each. Record yourself once a week. |

Plus **15 min LeetCode drill** (`drills/`), on any day you can.

Plus, **once a week, 20 min reading slot** (`reading/`): one real file (LeRobot, your harness), read only,
10-line note. That is the skill you need to check Claude Code's work on the Orin.

On 2-hour days: delete your solution and redo the lab from a blank file, no notes.
Then read the knowledge primer for the week (`knowledge/`).

## Rules

1. **No AI in exercise files.** No Copilot, no Claude Code autocomplete, no pasting.
   The repo's `CLAUDE.md` puts Claude in tutor mode: hints and questions only.
   If you are stuck for 10 minutes, ask for a hint, not code.
2. **Predict before you run.** Always. This is the fastest way to build intuition.
3. **Shapes in comments.** Every tensor line gets a `# (B, T, D)` comment.
4. **Solutions are on the `solutions` branch.** Look only after the tests are green,
   or after 20 minutes truly stuck. `git diff master solutions -- <file>` shows one file.
5. **Tick `PROGRESS.md`** at the end of each session. Honest marks only.

## Map

- `CURRICULUM.md`: the 9 weeks, day by day (all written)
- `weeks/`: the labs
- `drills/`: LeetCode-style problems, robotics-flavored, with tests
- `knowledge/`: dense primers (VLM, VLA, JEPA, world models, diffusion, eval) with
  interview questions. This is the research-discussion round.
- `reference/`: your own book, paper drafts and register (snapshot from `road-to-robotics`).
  Labs and primers link into it. The book's live-coding list (`exercises/coding/README.md`)
  is what these labs make runnable, with tests.
- `reading/`: the weekly read-only slot (protocol, template, one real file per week)
- `scripts/sync_reference.sh`: refresh `reference/` from your road-to-robotics clone
- `PROGRESS.md`: your tracker

## Why these three skills

| Skill | Interview round it trains | Daily-life use |
|---|---|---|
| Build | "Implement X in 45 minutes" | Write the code instead of delegating it |
| Predict | "What happens if…", whiteboard reasoning | Know what a result should look like before it arrives |
| Review | "This training script is broken, fix it" | Catch Claude Code when it is wrong |
