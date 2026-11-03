# Questions (out loud): read reference/journal/papers/jepa-world-model/AUDIT_W1_LOADS.md first

1. Tell the story of your checkpoint-loading bug in 90 seconds: symptom, how it was found, what it invalidated, what you changed.
2. Why is `strict=False` dangerous, and when is it legitimate? What do you check when you use it?
3. Which of your register rows depended on loads that were not counted? How did you decide which results to re-run?
4. The review bug: what would the training curve of that "fine-tune" look like? Name the 10-second check.
5. Normalisation buffers: what happens if they are not in the checkpoint, and how does it show up on the robot?
6. Interviewer: "Make this impossible to happen again on the team." (Tools, tests, process.)
