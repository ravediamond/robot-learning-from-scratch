# How to learn with this repo (evidence, November 2026)

This repo is a skill-formation program run by someone who also uses AI all day at work. The research
from 2025–2026 is unusually direct about that situation: **AI can raise what you produce while lowering what
you learn**, and the difference depends almost entirely on *how* you use it. These rules turn that evidence
into the daily protocol. Each rule says what to do, why, and where it is built into the repo.

## The evidence in one table

| Finding | Study | Numbers |
|---|---|---|
| Developers learning a new library with an AI assistant learned less, especially debugging | Shen & Tamkin (Anthropic), *How AI Impacts Skill Formation*, 2026, RCT, n=52 | quiz −17% (d = 0.74), largest gap on debugging; no significant speed gain |
| How you use AI decides the outcome | same study, screen-recording analysis | "conceptual inquiry" users scored 86%; full delegation and iterative AI debugging scored 24–39% |
| Answer-giving AI raises practice scores and lowers exam scores; a hint-only tutor does not | Bastani et al., PNAS 2025, ~1,000 students | practice +48%, exam −17%; tutor mode practice +127%, no exam harm |
| AI-assisted performance hides weak recall and ownership | Bergh et al., UNSW, 2026, programming RCT | task score 89% vs 69%, recall 41% vs 53% (48 h later: 39% vs 52%); "this code is mine" 45% vs 81% |
| Learners feel they learned more with ChatGPT and retain less | RCT, Brazil, 2026 (via Hendrick's May 2026 digest) | 45-day retention 57.5% vs 68.5% (d = 0.68): "borrowed competence" |
| Same AI access, two modes: substitute (no skill) vs complement (faster skill); AI-free gates reveal which | Yao, *When the Scaffold Stays On*, 2026, 10,419 competitive programmers | among proctored-contest qualifiers, more AI-style practice predicted *higher* unaided scores |
| Attempting a problem before asking AI improves learning | Akgun & Toker, *Struggle First, Prompt Later*, 2025 (3 experiments) | attempt-first beat AI-first; adaptive pre-questions helped most on complex tasks |
| Structured spaced retrieval beats open-ended AI study over weeks | Akgun & Toker, 2026, 7-week retention study | posttest 78.2 (adaptive spaced retrieval) vs 67.3 (learner-directed AI study), d = 0.92 |
| Spacing and retrieval are among the most robust effects in learning science | Carpenter, Pan & Butler, *Nature Reviews Psychology*, 2022 | decades of replications; their combination is strongest |
| Interleaving alone hurt long-term learning of complex content | Upper-secondary physics, n=376 (May 2026 digest) | helped only combined with collaboration |
| Worked examples help novices; extra prompts can overload | 2023 meta-analysis (May 2026 digest) | g = 0.48 |
| Metacognitive feedback beats encouragement for transfer | fNIRS study, n=87 (May 2026 digest) | transfer 9.38 vs 6.72 |

Honest caveats: most studies are short and use students; few test experienced engineers over months.
Treat these as the best current bets, and measure yourself (rules 3 and 4 produce your own data).

## The rules

### 1. Attempt before help (pretesting, productive struggle)
Open `predict.md` and the build file's signatures **before** reading the README, and write guesses. Wrong
guesses are fine: unsuccessful attempts still make the following explanation stick better. When a test fails,
debug **alone for 10 minutes** before asking for a hint. In Shen & Tamkin, the no-AI group hit more errors,
and those errors are where they learned the debugging that the AI group never learned.
*Built in:* day table (cold attempt first), `CLAUDE.md` (the tutor asks for your hypothesis before any hint).

### 2. Use AI as a complement, never a substitute
Allowed: conceptual questions ("why does Raft need the term check?"), explanations of code **you** wrote,
hint ladders, "explain this after I've tried". Not allowed on exercises: asking for the code, pasting errors and
applying fixes ("iterative AI debugging"), letting it write the solution. That is the difference between 86% and
24–39% in the RCT, and between Bastani's tutor (+127%, no harm) and the answer-giver (−17% on the exam).
*Built in:* `CLAUDE.md` tutor mode lists the allowed and forbidden modes; the weekly retro asks for your
honest delegation estimate.

### 3. Performance is not learning: test yourself later, without AI
Green tests today measure what you produced, not what you will know next week. Two AI-off checks:
- **Monday delayed recall (10 min):** from a blank file, rewrite the main function of last week's hardest lab
  (your pick, written down on Friday) with no notes, then run its tests. Delayed (3+ days), unaided, generative: the strongest signal available.
- **Friday AI-off gate (15 min):** one drill under a timer, AI fully off. If your lab marks are ✅ but gate
  scores fall, you are drifting into substitute mode (Yao's finding, applied to yourself).
*Built in:* `PROGRESS.md` weekly rows.

### 4. Spaced retrieval with feedback, every day
Once, after cloning: `git fetch origin solutions:solutions` (the tool reads model answers from there).
After each lab: `python scripts/review.py add <lab dir>`. Each day, start with `python scripts/review.py`
(5 minutes, up to 6 cards). Answer out loud **before** seeing the model answer, then grade honestly. Cards come
back after 1, 3, 7, 21, 60 days. Structured retrieval beat open-ended AI study by d = 0.92 over seven weeks.
*Built in:* `scripts/review.py`, `review/cards.json` (commit it: it is your learning record).

### 5. Block new material, interleave in review
Each week stays on one topic (new schemas need blocking). Mixing happens where it helps: review sessions
shuffle cards across all past labs, and drills mix patterns from earlier weeks.
*Built in:* curriculum order, `review.py` shuffles.

### 6. Worked example, then rebuild
If you are truly stuck after 20 minutes on a first exposure, study the solution as a **worked example** (read
it, explain each line out loud), then close it and rebuild from a blank file **within 48 hours**. Mark it ❌
honestly; the rebuild is what counts.

### 7. Feedback that makes you think, not feel good
Ask the tutor for "what did you expect, what happened, why", not reassurance. Write the one-line `WHY:` in
every review file and the misses line in every `predict.md`: explaining in your own words builds ownership,
and ownership tracked retention in the UNSW study.

## Daily budget (1 hour)

| Min | Step |
|---|---|
| 0–5 | `review.py` (spaced retrieval). Mondays: the 10-minute delayed recall instead, starting at 0. |
| 5–10 | Cold attempt: predict.md guesses before reading anything |
| 10–18 | Read the README |
| 18–20 | Run `predict_run.py`, write misses |
| 20–45 | Build (AI off; 10-minute solo debugging before any hint) |
| 45–53 | Review file (find the bug, write WHY) |
| 53–60 | Questions out loud, then `review.py add <lab dir>` |

## Sources
- Shen & Tamkin, *How AI Impacts Skill Formation* (2026): https://arxiv.org/abs/2601.20245
- Yao, *When the Scaffold Stays On: AI, Practice Style, and Screening in Elite Skill Formation* (2026): https://arxiv.org/abs/2606.06253
- Bastani et al., *Generative AI without guardrails can harm learning*, PNAS 122 (2025): https://papers.ssrn.com/abstract=4895486
- Bergh, Tag, Vassar, Renzella, *Your Programming Students' Cognition with ChatGPT: Higher Performance, Lower Retention, and Reduced Ownership* (2026): https://arxiv.org/abs/2609.21194
- Akgun & Toker, *Struggle First, Prompt Later* (2025): https://arxiv.org/abs/2504.10249
- Akgun & Toker, *Do Gains from Generative AI–Enabled Adaptive Pretesting Persist?* (2026): https://arxiv.org/abs/2606.22328
- Carpenter, Pan & Butler, *The science of effective learning with spacing and retrieval practice*, Nature Reviews Psychology (2022): https://www.nature.com/articles/s44159-022-00089-1
- Hendrick, *What's New in Learning Science, May 2026* (digest of the interleaving, worked-example, feedback and ChatGPT-retention studies): https://carlhendrick.substack.com/p/the-monthly-dispatch-whats-new-in-3c4
