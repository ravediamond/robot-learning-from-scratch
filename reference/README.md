# Reference: your own book, papers and register

Snapshot copied from `road-to-robotics` at commit `2c24b56` (2026-11-03). Read-only here:
the source of truth stays in `road-to-robotics`. Re-sync by copying `journal/book`,
`journal/papers` and the four register files again.

| Path | What | When to use it |
|---|---|---|
| `journal/book/` | "Evaluating Robot Learning on Cheap Hardware", the full book | Each lab README and knowledge primer links to the matching chapter |
| `journal/book/00p1-deep-learning-from-zero.md` | Chapter P1 | Week 1, before d3 (autograd) |
| `journal/book/00p2-pytorch-for-the-coding-round.md` | Chapter P2 | Week 1 to 3: the API reference for the labs |
| `journal/book/exercises/coding/README.md` | The 26-item live-coding list | The labs implement these items with tests. Item numbers are cited in each lab |
| `journal/book/exercises/interview/` | Opener, pushback drills, fault cards | Week 8 mocks |
| `journal/papers/` | Your paper drafts, feasibility memos, reviews | Week 8 (defend your own work); any day you want a worked example of the method |
| `journal/EXPERIMENTS.md`, `LESSONS.md`, `DECISIONS.md`, `METHOD.md` | The register and the method | Week 6 (evaluation), week 8 (re-derive your numbers by hand) |

Not copied: `TIMELINE.md`, `strategy/`, prompts and workstack (personal or operational).
Some links inside the book point to files outside this snapshot; those are expected to break.

**Rule for week 8:** every number you have published must be re-derivable by you, from
the raw data, with your own code. If you cannot, do not cite it in an interview.
