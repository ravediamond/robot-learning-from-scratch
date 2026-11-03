# W9 D4: Research discussion mock: paper → experiment → verdict

Book: 1 (start from a decision), 2/2a (translate the claim, read a paper), 3, 4 (pre-registration),
5 (one factor, three verdicts, the confound checklist), 7. This is the round where your eval background
should make you stand out. Do it out loud, recorded.

## Part A (30 min, written): `paper_to_experiment.md`
Pick ONE claim from a paper in your KB (suggestions in the file). Fill the template BEFORE looking at any
number: decision, claim in testable form, cheapest baseline, metric, n and why, pass line, fail line,
verdict map (which confounds turn a result into "not proven"), cost.

## Part B (20 min, code): your verdict rule as a function
Your book's rule (5.4–5.6): a pass line P and a separate fail line F (F < P) on the improvement d with
its 95% CI [lo, hi]:
- **pass**: d ≥ P and lo > 0
- **fail**: d ≤ F and hi < P
- **not proven**: everything else, and ALWAYS when an unremovable confound is listed.
Then run it on the mock experiment (`mock_experiment.py`: a "+15 points" claim, 30 paired starts).

## Today's build (`w9d4_verdict.py`)
`paired_diff_ci`, `verdict`.

Run: `uv run pytest weeks/w9-mock-interviews/d4-research-verdict -s`
