# Chapter 9. The register

## 9.1 The principle

Every experiment gets one row, written once, never edited, in a file that is the single place anyone looks to find out what was tried and what it showed. Lessons get one line each with the evidence named. Decisions get one row with their status. Corrections are new rows. Chat-only facts are marked provisional until a file records them.

This is the least glamorous chapter and the one whose absence would have made every other chapter impossible to write. The method is only as good as the memory of what it produced.

## 9.2 Why: what goes wrong without it

Three failures from three weeks, each found *because* the register existed and would have been invisible without it:

- **"G9b running since 10-16."** The line was in NEXT_ANGLES and in EXPERIMENTS §3 for a week. The job had died at preprocessing on 10-16 and the instance had self-destroyed. The register's own rule (every claim links to a file) is what made the check possible: the linked report showed only a zero-shot baseline. The write-up that was supposed to summarise G9b became a re-analysis that found a second error (400 teleop demos counted as successes in the first summary).
- **The D-JEPA description.** NEXT_ANGLES §1 described a loss term. The paper describes an operator. Because the protocol had to cite the paper's section before running, the mismatch was caught at protocol-writing time, and the amendment is on record. Without the "protocol cites the paper" rule, the wrong experiment would have run and its null result would have been filed as a refutation of D-JEPA.
- **The TurboVLA port.** The conclusion was published; the comparison to upstream came two weeks later. The register's row says "closed: ignores the instruction"; the correction row will say "port diverged; faithful reproduction: [result]". Both rows stay. A reader sees the sequence, which is itself informative: this is what careful work looks like when it is wrong.

Without a register these become: a line in a chat that nobody re-reads; a result someone half-remembers; a blog post that is quietly wrong.

## 9.3 The files and what goes in each

| File | One entry per | Fields | Never |
|---|---|---|---|
| `EXPERIMENTS.md` | gate or measurement | date, ID, question, bar (pre-registered), result, verdict, cost, source link | edit a row; file a number without its source; use *pass*/*fail* for a measurement |
| `DECISIONS.md` | decision | date, ID, decision, reason, status (recorded / chat / open), the experimenter's standing rules | mark a chat decision as recorded before the file exists |
| `LESSONS.md` | lesson | the lesson in one sentence, the evidence (gate IDs and numbers) | a lesson without evidence; a lesson that restates a result |
| `TIMELINE.md` | day | what happened, where it is recorded | narrative |
| `ARCHITECTURE_HISTORY.md` + `architectures/` | version | one line in the table; verbatim snapshot of the outgoing spec | rewriting history when a version is superseded |
| `PAPERS.md` | paper | what it changed here | a paper that changed nothing |
| `NEXT_ANGLES.md`, `PREP_QUEUE_*.md` | planned experiment | question, decision, existing assets, steps, bar, cost, prompt | a plan whose bar is blank at run time |
| `drafts/` | write-up | the 8-section template | a draft that is not backed by a register row |

The source column is load-bearing. Every number in a row must be in a committed file (a report README, a results JSON, a protocol). "chat" as a source means the number exists only in a conversation and is provisional. The 10-22 audit that found G9b never ran was done by following source links.

## 9.3a The claims layer and the contradiction table

Since WS-061 (2026-10-30) the register has a layer above the gate rows: per paper, its goal, its hypotheses and its tested claims as subject / relation / object / metric / conditions rows (chapter 2.6a). Three rules carry over from the gate rows. Each claim row cites where in the paper it is stated (section, table, figure). Each "ours" number names its source (a folder's check, a JSON path, a CSV) and the build re-reads it and fails on a difference. Contradictions are computed, never typed: the five listed in 2.6a are what the build found, and a hoped-for contradiction that the data did not support (ReAP's action conditioning) was not forced into the table.

The paper verdict is derived from the paper-stated tested claims only; rows marked "ours" never change it. That is why PACT-VLA reads *fail* (its stated "no added delay" fails) while its offline selection result, which replicates, keeps its own gate rows.

## 9.4 Correction rows

When a result is found wrong, the original row stays and a correction row is added with the same ID plus a suffix and a one-line explanation:

> | 2026-10-18 | G11-A re-check | Did G11-A run the trained W1? | — | No: `strict=False` + key remap under a different transformers version; encoder random. Timing valid, latents void. | correction | free | [link] |

Why not edit: the original row records what was believed at the time and what decisions were taken on it (v0.8.6 was written around G10 and G11-A). Editing it would make those decisions look unmotivated. The sequence is the evidence that the method catches its own errors, which is the thing an interviewer most wants to see.

## 9.5 Lessons: the form

A lesson is a sentence that stops the next person re-learning it, with the evidence that earned it. The 23 in `LESSONS.md` follow a shape: *claim* (one sentence, general), *evidence* (gate IDs and numbers), sometimes *consequence* (what changed). Example:

> **A world model that predicts well can still be useless for choosing actions.** G0 passed (K=30 error 0.31 vs copy-last 0.89; reacts to action timing), W2 encodes the arm (R² 0.92), RoboJEPA uses our actions (9–10 % below shuffled). Yet G1b failed and an action-only scorer beat both world models. Prediction quality is not decision quality.

A lesson that could not be written this way (claim + evidence) is an opinion; file it in DECISIONS as a standing rule if it is one, or nowhere.

Lesson 24 (closed 2026-10-23, W2′ and W1 on the Hub; LESSONS.md): *every checkpoint a frozen protocol depends on goes to the Hub the day it is produced.* Evidence: W2 lost with a destroyed vast instance (10-22); G7a blocked; W2′ retrain $1.25 and a day.

## 9.6 What the register is not

It is not a scoreboard. Passes and fails are not points. Three weeks produced roughly six passes and twenty fails or not-provens, and the fails are where most of the knowledge is (chapter 8). A register that made fails look bad would be edited; this one makes them look like what they are: measured.

It is not a narrative. The narrative is in the write-ups and on the site. The register is the index the narrative links to.

It is not finished. Rows get correction rows; lessons get superseded (lesson 7's "judging is easier than imagining" has a note: "on the complete set Robometer fails zero-shot (0.818), so the 'passed' above held only for the partial set"). The note is the right form: the original stays, the update is appended, dated.

## 9.7 Keeping it current

After every experiment: one row in EXPERIMENTS (bar, result, verdict, cost, link), one line in TIMELINE. After every decision: one row in DECISIONS; when the decision is written into a file, switch its status. Architecture change: snapshot the outgoing version, add a line to the history table. Every week: read EXPERIMENTS §3 (pending) against the files it links to and check that each pending item is actually pending. That weekly read is what found G9b.

The tools do the bookkeeping well and the audit badly: they write rows at the speed they produce results and do not notice that a row contradicts a file written two days earlier. The audit is yours.

## 9.8 What the interviewer will ask

*"How do you keep track of experiments?"* The files, the one-row rule, the source link, the correction rows. Then the G9b story: "the register is what let me find that a job I'd been describing as running had died a week earlier."

*"How would you set this up for a team?"* Same structure, with the protocol commit as the gate to compute access, and the weekly pending-audit as a rotating duty. The point to make: the register is cheap (one row) and the audit is where the value is.

*"Show me a result you changed your mind about."* Any correction row. Tell it as a sequence: what was believed, what was decided on it, what was found, what changed.

## 9.9 Exercises

1. Audit `EXPERIMENTS.md` §3 (pending) today: for each row, open the linked protocol and confirm it is pending, running, or done. File corrections.
2. Write the correction row for TurboVLA now, with a blank for path A's number, so the shape is ready.
3. Rewrite lessons 1–7 in your own words without looking, then compare. Where your version has no evidence, find it or drop the lesson.
4. Write lesson 24 and file it.
