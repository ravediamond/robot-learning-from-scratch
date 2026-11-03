# W8 D5: Checkpoint forensics: never trust a load you did not count

Your audit: `reference/journal/papers/jepa-world-model/AUDIT_W1_LOADS.md`. Book: 6 (trust the
instrument), 16.1c (the checkpoint-loading bug in the interview). Coding list item 22.

## How loads fail silently
- `load_state_dict(sd, strict=False)` returns `(missing_keys, unexpected_keys)`: code that ignores the
  return value loads NOTHING when the prefixes differ, and nobody notices.
- Prefixes: DDP saves `module.layer...`, a wrapper saves `model.layer...`, LeRobot policies nest submodules.
- Renamed layers between versions (`head` → `action_head`).
- Shape mismatches (a different action dim or chunk size).
- Buffers (normalisation stats!) that are not saved, or are re-initialised after loading.

## The checklist (make it a function)
1. Count: tensors expected, loaded, missing, unexpected, shape-mismatched.
2. Fail loudly unless the count is exactly what you expect.
3. Verify the CONTENT: weights equal to the file; outputs on a fixed input equal to the outputs recorded
   at save time.

## Today's build (`w8d5_forensics.py`)
`load_report`, `remap_keys`, `safe_load`, `fraction_changed`.
Then read your AUDIT_W1_LOADS.md and answer the questions.

Run: `uv run pytest weeks/w8-robotics-register/d5-checkpoint-forensics`
