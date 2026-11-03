# LeetCode drills (15 min each)

Algorithm rounds exist at many robotics startups, usually one round, medium level. You do
not need 300 problems. You need ~15 **patterns**, recognized fast and coded clean.

## Protocol
1. Start a 15-min timer. Read the docstring. Say the pattern and the complexity OUT LOUD
   before typing ("hash map, O(n) time, O(n) space").
2. Write it. Run `uv run pytest drills/w1 -k p01`.
3. Over 15 min: stop, look at the pattern note in `PATTERNS.md`, redo it tomorrow from blank.
4. Log time + one-line pattern in `PROGRESS.md`.

Interviewers score: talking while coding, edge cases stated before running (empty input,
one element, duplicates, negatives), complexity stated without being asked.

## Week 1 patterns
| Drill | Pattern | Robotics flavor |
|---|---|---|
| p01 | Hash map | find two sensor readings that sum to a calibration target |
| p02 | Stack | validate nested command blocks |
| p03 | Sliding window + monotonic deque | max torque over the last k control steps |
| p04 | Sort + merge intervals | merge overlapping failure segments of a rollout |
| p05 | Binary search on a monotone predicate | first frame where the gripper is closed / first bad checkpoint |
