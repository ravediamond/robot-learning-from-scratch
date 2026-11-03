# W9 D3: "This training script is broken. Fix it." (45 minutes)

The debugging round. `broken_bc.py` trains a chunked BC policy on the obstacle-world demos and saves a
checkpoint for deployment. It runs end to end without an error. It contains **7 planted bugs**, all of
kinds you met in this course. Some are in data handling, some in training, some in evaluation and saving.

## Protocol
1. Timer on (45 min). Read the whole script first; say out loud what each function should do.
2. For each bug: name it, fix it, and write one line in `FIXES.md` (what it was, how it would show up).
3. `uv run pytest weeks/w9-mock-interviews/d3-fix-the-script` has one test per bug; the test names do NOT
   say which bug they target. Aim to find bugs by reading, then use tests to confirm.
4. Count how many you found before running the tests. That is your interview score.
