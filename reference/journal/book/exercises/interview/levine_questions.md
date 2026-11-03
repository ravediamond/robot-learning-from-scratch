# Six questions Levine's talk implies (16.1c), answered from this rig

Each answer is ≤ 90 words; the long form is in the chapter named after it.

**1. The connector question: "What is the lowest level you have gone to when a result was wrong?"**
The loader and the register file. G11-A looked like a world-model result; `strict=False` and a key remap had left the encoder at random init, and a strict-load check found 192 missing keys. A scoreboard showed one failed gate too many; unescaped pipes had shifted a row's cells. Below that, five source builds to get LeRobot onto the Orin. Each time I fixed the layer and re-ran before touching the model. *(16.1c, 8.7, chapter 6.2–6.3.)*

**2. "What is the robotics analogue of prompt expansion?"**
Richer annotations that describe the behaviour. My data measures the problem: on ~94 % of the bowls frames the instruction carried no information about the action — the pose already decided the side — so even the authors' own TurboVLA trainer ignored it (0.020°). The first thing I would add is sub-task annotation ("grasp the blue bowl", "place in pink"), tested by selected-versus-random training, with the chunk-boundary cost of shorter segments stated. I have not trained with them yet. *(16.5a, chapter 11.4, chapter 13.)*

**3. "Interventions fix mistakes; RL avoids them. Where are you on that path?"**
At the first two steps in miniature: corrections in the tidy-loop teleop set, failures labelled and kept out of training, a harness that records why a run stopped. The missing piece is the reward: the only reliable one here is the deterministic judge at rest (zero false positives in 25), and learned judges did not hold — Robometer-4B failed as a zero-shot judge, the Gemini judge fitted for G16 caught 0/8. No RL on hardware yet; I say that first. *(16.5a, chapter 14a.)*

**4. "Why solve all of robotics instead of one humanoid task?"**
Because generality is where the edge cases accumulate, and a general model sees each one more often. On one task my failures concentrated: C1's failures belonged to one policy, and policy identity alone beat every probe. A narrow setup hides its own confounds. The low-cost version of generality is many papers on one rig, which is what the register is: fifteen papers, 73 gates. *(16.1c, chapter 5.)*

**5. "Cars are too specialised — why does that matter for bootstrapping?"**
A specialised deployment generates data only for its own distribution. Mine shows the cost in small: thresholds fit on seen rollouts flagged 42/42 fresh successes (W1-A), and a monitor's 0.93 fell to 0.79 once episode length stopped leaking the outcome. Bootstrapping needs a deployment broad enough that its own data teaches the next model something new, and a judge you trust on it. *(Chapter 8.5, chapter 13.)*

**6. "Wheels or humanoid for the home?"**
Wheels and arms first, because the expensive part is not locomotion but reliable manipulation and judging it. On a $230 arm the binding constraints were contact sensing (none), reach, and an on-board computer that could not run REKI's kernels — the model was served from a rented GPU. A humanoid adds balance and cost before any of those are solved. I would want manipulation at rest to be judged reliably first. *(Chapter 10, the rig page.)*
