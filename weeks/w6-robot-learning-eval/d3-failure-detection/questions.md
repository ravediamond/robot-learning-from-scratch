# Questions (out loud)

1. Define AUROC in one sentence a product manager understands. Then say why it is not a deployment decision.
2. Why do ties matter for episode-length scores in particular? What did the review bug do with them?
3. Conformal threshold: what is guaranteed, under which assumption, and what is NOT guaranteed (think: failures)?
4. Your length-null result on your own rollouts (length_null_real.py). Does your W1 world-model monitor (0.84 [0.73, 0.93]) beat it? What is the right test for "beats", given they score the same episodes?
5. A monitor fires at the moment of failure with AUROC 0.95. Why can it still be useless for recovery?
6. Interviewer: "Design the evaluation of a failure monitor for a fleet of 10 arms." Data, baselines, metrics, bars.
