# W12 D5: reproducibility and the report + design

> **Full lesson for this day:** [`../README.md`](../README.md#day-5-reproducibility-the-report-a-design) (the week page: concepts from zero, examples, traps).
> This file is the one-page summary. Order of work: `predict.md` → the lesson → `predict_run.py` → the exercise → the review file → `questions.md`.

Primer: `knowledge/13-evaluation-at-scale.md` §5–6.

## The one idea

A result you cannot reproduce in six months is an anecdote. Make every number traceable:

- a **fingerprint** of the exact weights: SHA-256 over the sorted parameter names, dtypes, shapes and
  raw bytes (not a file name, not `str(state_dict)`: PyTorch abbreviates big tensors with `...`);
- a **run manifest** written at the start of every session: git SHA, checkpoint fingerprint, dataset
  version, config hash, hardware, package versions;
- a **manifest diff** to answer "why did these two runs give different numbers?" in one command;
- a **results table** generated from the records, never typed by hand.

Then **design #1** of this repo: an evaluation program for a generalist manipulation policy.

## Today's build (`w12d5_report.py`)

`tensor_fingerprint`, `config_hash`, `run_manifest`, `flatten`, `manifest_diff`, `results_table`.

## Staff angle

"Make robot learning reproducible, measurable and deployable" is a pitch you can only make if your own
results carry a manifest. Put the manifest next to every table on your project page.
