# 10. Auditing what Claude Code did on the Orin: a map of road-to-robotics

Mapped from `road-to-robotics` at commit `c2a37e5` (2026-11-03). The harness reference is
`docs/HARNESS.md` (296 lines, verified 2026-11-01) and the worker's rules are
`.claude/skills/workstack/SKILL.md`. This page does not repeat them. It answers one question:
**given a result Claude Code produced, where do I look to check it, in what order?**

## 1. The trail of one item (WS-NNN)
| Step | Where | What to check |
|---|---|---|
| 1. The ask | `journal/workstack/QUEUE.md` item (or `ARCHIVE.md` once done); `python3 scripts/ws.py show WS-NNN` | `why:`, `needs`, `kind`, `model`: what was asked and why |
| 2. The brief | `prompt:` → `journal/prompts/<date>_<slug>.md` (≤ 25 lines) | the deliverables and the gate; is the result answering THIS? |
| 3. The bar | the project's `PROTOCOL.md` (e.g. `projects/<name>/PROTOCOL.md`) | **committed before any number?** `git log --format='%h %ad' -- PROTOCOL.md` vs the result commit |
| 4. The work | commits: `git log --oneline --grep WS-NNN`; diffs: `git show <hash> --stat` | what changed, in which files, in what order |
| 5. Notes and logs | `journal/workstack/notes/WS-NNN.md`; `log:` lines on the item | numbers with their files, paths created, open questions |
| 6. Raw output | `journal/workstack/runs/<WS>/` (full logs written by `scripts/cc/cc.py`; `trajectory.jsonl` from the hooks, gitignored, local to the Orin) | the actual command output behind each number |
| 7. The result | `result:` on the item: `path — number, n, verdict` | the path exists; the number is in it |
| 8. The register | `journal/EXPERIMENTS.md` row; the project README | same number, same verdict wording, link back |
| 9. Gates you typed | `journal/workstack/GO.txt` | every arm/vast/post action has a matching `go` line |

## 2. What `ws verify` does and does NOT check
`scripts/ws.py: cmd_verify` (line 741) refuses a `result:` that has numbers without a file path, or numbers
absent from every cited file. That is all. It does **not** check that the number was computed correctly, from
the right data, with the right split, on a loaded checkpoint. "verify passed" = "the number is written
somewhere", not "the number is right". Your job is steps 3 and 6 above.

Numbers tagged `unsourced:` are never citable (SKILL §3.5). Facts (dataset ids, revisions, counts) should come
from `data/FACTS.json` (built by `scripts/facts_build.py`), not from memory.

## 3. Map of the repo for reading
| Path | Holds |
|---|---|
| `journal/README.md` | index + "current state". **Dated**: check the date before trusting a "current" line |
| `journal/EXPERIMENTS.md` | the register: one row per gate (question, bar, result, verdict, cost, link) |
| `journal/DECISIONS.md`, `LESSONS.md`, `ARCHITECTURE_HISTORY.md` | why things changed |
| `journal/papers/<slug>/` | memos, reviews, audits (e.g. `jepa-world-model/AUDIT_W1_LOADS.md`); paper drafts live in `ravediamond/paper-*` |
| `journal/harness/` | the harness's own measurements, bench, ACCEPTANCE, LESSONS, changelog |
| `projects/<name>/` | one folder per experiment: README, `PROTOCOL.md`, scripts, reports |
| `projects/vla-jepa-world-model/wm_train/` | LeWM training (`train_lewm.py`, `train_w2.py`, `wm_report.py`) |
| `agent/policies/rtc_controller.py` | the RealTimeChunker (async execution) |
| `shared/gate/` | gate scripts and their data |
| `scripts/ws.py`, `scripts/cc/cc.py`, `scripts/autopilot.py` | queue CLI, output wrapper, autopilot |
| `.claude/hooks/*.py`, `.claude/settings.json` | the rules as code (`pre_bash.py` blocks rm, sudo, posting, vast without go) |
| `.claude/agents/*.md` | subagents (reader, number-checker, protocol-readback, site-checker, trajectory-reviewer) |
| `strategy/` | private judgements: not part of any audit |

## 4. Failure patterns already seen (from HARNESS.md §14 and the register)
Check these first, they happened:
1. **Silent partial load**: `strict=False` + a key remap dropped 192 encoder tensors: a random encoder
   (AUDIT_W1_LOADS.md). Check: load report, tensor count.
2. **Typed numbers**: label counts 46/23 vs 44/25 typed by hand. Check: the number comes from a command over
   the file, in the same turn.
3. **Retrospective evaluation**: G10's AUROC 1.000 on seen rollouts with thresholds fit on the same data
   (book 5.6). Check: are the evaluation episodes fresh, and was anything (thresholds, stats, probes) fit on them?
4. **Wrong implementation tested**: TurboVLA port vs upstream (book 5.6a). Check: parity against reference code.
5. **n too small to conclude**: 2 runs per side gave a 35 % swing (harness LESSONS B2). Check: n, seeds, CI.
6. **Pandas column named `T`** shadowing the transpose (ANALYSIS_2026-11-01). Check: suspicious analysis code.

## 5. A 20-minute audit routine (use in the weekly reading slot)
1. Pick a `done` item from `ARCHIVE.md` that produced a number you cite.
2. Steps 1–3 of the trail: was the bar frozen first?
3. Open the script that produced the number. Write the reading template (`reading/TEMPLATE.md`).
4. Re-run the number from the raw file if cheap (as in W8 D4). Otherwise find it in `runs/<WS>/`.
5. Verdict in one line: **stands / stands with a caveat / does not stand**, and why. If it does not stand,
   `python3 scripts/ws.py correct WS-NNN "what"`: corrections are the only accuracy signal the harness
   loop cannot produce itself (HARNESS.md §9).

## Questions
1. Name the three places a number must agree before you cite it in an interview.
2. Why is "the PROTOCOL.md commit is older than the result commit" a check you can do in 10 seconds, and what does it prove?
3. What can `trajectory.jsonl` tell you that the commits cannot? Where does it live?
4. Pick one item from last week. Run the routine in §5. What did you find?
