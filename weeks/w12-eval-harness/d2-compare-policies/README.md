# W12 D2: comparing several policies honestly

Primer: `knowledge/13-evaluation-at-scale.md` §2. You wrote Wilson, McNemar and the paired bootstrap in W6 D2.

## The one idea

Four policies make **six** pairwise comparisons. Test each at α = 0.05 and the chance of at least one
false "significant" result is close to 26%. **Holm** keeps the family-wise error at 5% and is never
worse than Bonferroni: sort the p-values, multiply the smallest by m, the next by m−1, …, and make the
sequence non-decreasing.

The question people actually ask is "which one is best?". Answer it with uncertainty: resample the
**start configurations** (all policies together, so the pairing survives) and count how often each
policy comes out on top: **rank probabilities**.

And before any of that: what can 50 rollouts per policy detect at all? The **minimum detectable
difference** (80% power, α = 0.05) around 50% success is about 27 points. A 10-point gap needs
hundreds of rollouts per arm. Pairing helps, but say this number before showing a table.

## Today's build (`w12d2_compare.py`)

`mcnemar_p`, `holm`, `pairwise_table`, `rank_probabilities`, `min_detectable_difference`.

## Staff angle

"A beats B, p = 0.04" across six comparisons with no correction and 50 rollouts is the most common
over-claim in robot-learning papers. Saying "with this n we can only see differences above 27 points;
here is what we can and cannot conclude" is what a research lead wants from the person who owns evals.
