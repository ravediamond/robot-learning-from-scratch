# Chapter F. The fast track: learning this with little time

This book is about 100 000 words. Most readers of a book about a one-person lab are running that lab beside a job, which means a few focused hours a week at best. Read cover to cover, the book takes months; and reading is the slowest way to acquire what it teaches, because what an interviewer, a reviewer or a collaborator tests is what you can **produce** without notes: code written cold, a result explained with its number, its n and its baseline, a paper taken apart in twenty minutes.

This chapter is the path for that situation. It cuts the scope to what is tested, replaces reading with practice, and fits in about 45 minutes a day plus one hour at the weekend.

## F.1 The principle: produce, do not re-read

Three findings from the learning literature carry the whole plan, and each one is a habit, not a tool:

1. **Retrieval beats re-reading.** Recalling an answer from memory (writing the code, saying the number) strengthens it far more than reading it again. Re-reading feels productive because the text becomes familiar; familiarity is not recall.
2. **Generation beats recognition.** Writing an attention layer from an empty file teaches more than reading a correct one ten times. The coding list is built for this.
3. **Spacing beats massing.** Twenty minutes a day for five days beats two hours once. Short daily sessions are not a compromise forced by a busy schedule; they are the efficient format.

Corollary: every session ends with something produced (a file, a recording, a card answered), never with "read chapter X".

## F.2 The daily session (45 minutes, weekdays)

- **30 minutes — one coding item, cold.** Empty file, no documentation, no assistant, timer on. Then compare with the reference named in P2.17, fix, and note the time. Redo any item the next day if it took longer than its budget.
- **10 minutes — one result, out loud.** Pick one row from your register and say it in under a minute: the question, the number with its n, the baseline, the mechanism, what it changed. Record it on a phone; listen once.
- **5 minutes — flashcards** (F.4).

## F.3 The weekly session (one hour, weekend)

A mock interview with a language model in voice mode, one format per week, rotating:

- **Project deep-dive.** Instruction to the model: "You are interviewing me for a research engineer role on world models. I will present a project. Ask 'why' at least three levels deep on every number I give, interrupt me if I talk for more than two minutes, and at the end list every claim I could not justify." (Chapter 16.2a.)
- **Paper discussion.** Give it a paper you have not read in depth; take twenty minutes with the PDF; then have it ask the seven questions of 16.3a.
- **Fundamentals.** Have it quiz you from 16.4b and P1, one question at a time, and mark wrong answers for the flashcards.
- **Problem session.** One prompt from 16.5a, five minutes to answer aloud, then have it play a sceptical hiring manager.

Write down the three weakest answers after each mock. They are next week's daily-session material.

## F.4 Flashcards for what must come out without hesitation

Some things must be instantaneous in an interview: your own key numbers, the definitions, the fundamentals answers. These are exactly what spaced repetition (Anki or any equivalent) is for. Five minutes a day. Card types:

- **Your numbers.** Front: "G1b: world-model ranking vs commanded-joint baseline?" Back: "0.60 vs 0.68–0.90; n; mechanism: position control, decoys differ in arm state; Planning Limits." One card per register row you would put on a slide.
- **Definitions** from the vocabulary table (P1.17): front the term, back one sentence in your own words.
- **Fundamentals** from 16.4b: L1 vs L2, convexity, ELBO, bf16 vs fp16, Wilson at small n, why chunks.
- **Shapes and APIs** that trip people up: `nn.Linear` weight shape; the boolean mask convention in `nn.MultiheadAttention` vs `F.scaled_dot_product_attention`; what `model.eval()` does and does not do.

Write the cards yourself; making the card is half the learning.

**The faults deck.** One card per row of 16.1c's table and per fault named in 8.7: front, "presented as"; back, "lived at" plus the probe that found it. The cards are written out in `exercises/interview/faults_cards.md`, sorted by the eight layers of the public ledger (hardware, platform, engine, loader, data, model, evaluation, register).

## F.5 What to cut, and what never to cut

**Must do (the interview path):**

- Coding items **14, 23, 02, 03**, then **15, 11** (JEPA step, latent CEM, multi-head attention, training loop, SIGReg, flow matching). Items 14 and 23 first if the target role names JEPA or latent planning.
- Chapter 16: **16.1b** (how interviewers score), **16.2a** (deep-dive), **16.9c** (mistakes), **16.9b** (how to talk about the tools you used).
- Two stories, rehearsed until they fit in two minutes each: one planned result that failed with a mechanism, one correction of your own claim.
- Chapter 3 (the baseline) and chapter 7 (uncertainty): they are behind most follow-up questions.
- Chapter 11b.4 (the tick budget) and 2.6a (claims as rows, the five contradictions): the two newest results, and the PACT-VLA story is the shortest negative result to tell (16.9).
- Chapter 10a, sections 10a.1–10a.6 (frames, kinematics, actuators, control, the two clocks, action spaces), and its ten one-minute questions: "deep learning without robotics" is the most-cited red flag for robot-learning roles.

**Skim, only where a coding item points:** P1 and P2, through the map in P2.17. Read the section the day before the item, not the whole chapter.

**Skip until the above is solid:**

- The rest of Part I, if you already work in evaluation or experimentation professionally; much of it will be familiar, and chapters 3 and 7 carry the parts that are not.
- The "a" paper chapters, except the one closest to the target team's own work (for a JEPA world-model team, 12b: Planning Limits, LaLa-World, the planning study).
- Tier 3 of the coding list, unless the recruiter confirms an algorithmic round.

**Never cut:** writing code cold, and saying your numbers out loud. Everything else can be looked up; those two cannot be faked in the room.

**Papers.** Only Tier 1 of Appendix R (twelve papers, about 25 hours) is on the fast track, one per weekend, read with the 2a.8 template before opening the chapter that summarises it.

## F.6 Everything is in the book; outside resources are optional

Nothing outside this book is required. The two things that usually send people to long video courses are in chapter P2 as worked builds, each readable in one sitting and runnable on a laptop CPU:

- **P2.19, an autograd engine from scratch** (about sixty lines of plain Python): how `backward()` works, why gradients accumulate, why `zero_grad` exists, a small neural network trained by hand. About 45 minutes including running it.
- **P2.20, a tiny GPT end to end**: data, causal attention head, multi-head attention, MLP, pre-norm residual block, embeddings, cross-entropy, training loop, generation, and the sanity checks (initial loss ≈ ln V). About an hour; it trains in twenty seconds.

Read each once with the code running, then write it again from an empty file the next day. That is the same learning loop as the coding list.

**Optional, if you have spare time and like video:** Andrej Karpathy's free series *Neural Networks: Zero to Hero* builds the same two things in front of you (*micrograd*, then *Let's build GPT*), about two hours each at normal speed. It covers the same ground as P2.19–P2.20 at more length; it adds nothing the coding round needs that these sections do not.

## F.7 Protect the time

- **Apply before you feel ready.** A hiring loop takes three to ten weeks from first screen to offer; that is the preparation window, and a date forces focus. At the first call, ask what each round covers, then prepare only those rounds.
- **Freeze new experiments during preparation.** Let the automation finish what is queued; read only the one-paragraph results. A new result is worth adding to the story only if it changes a conclusion you will present.
- **One thing per session.** A 45-minute session that switches between a paper, a code item and the register produces nothing; pick one before you sit down.

## F.8 A four-week fast track

| Week | Daily (45 min) | Weekend (1 h) |
|---|---|---|
| 1 | P2.19 and P2.20 read and run (one sitting each); items 14, 23, 02, 03, each twice; the 16.2 opener with the 16.1c clause aloud (`exercises/interview/opener.md`) | Mock deep-dive #1; write the two stories; 16.7b as a dialogue (`exercises/interview/pushback_world_models.md`) |
| 2 | Items 15, 11, 12, 08; P2.19 rewritten cold | Mock paper discussion (16.3a) |
| 3 | Debugging drill; items 13, 16–19; 16.5a prompts aloud | Mock fundamentals quiz; fix flashcards |
| 4 | Redo the slowest five items; 10-slide talk (16.2b) | Full mock loop in one day; fix the three weakest answers |

Flashcards every day from week 1. When the coding items fit their time budgets and the mock deep-dive ends with no unjustified claim, stop preparing and go to the interview.
