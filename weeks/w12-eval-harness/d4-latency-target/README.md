# W12 D4: latency on the target, and what it does to chunked control

Primer: `knowledge/13-evaluation-at-scale.md` §4. Builds on W8 D3 (chunk switching).

## The one idea

A policy that runs at 30 Hz on your Mac may not on the Orin, and the mean hides what hurts: the tail.
Measure on the target, after warm-up (the first calls compile kernels and allocate memory), with the
real input pipeline, and report **p50 / p95 / p99 / max** and the **deadline-miss rate**. Percentiles
by **nearest rank** on the sorted samples, never "mean + 2σ": latency is not Gaussian.

Chunked policies change the budget:

- **Synchronous**: execute `e` actions of a chunk, stop, infer, continue. The arm is idle a fraction
  `L / (L + e·T)` of the time (T = control period). Visible as stutter.
- **Asynchronous**: infer the next chunk while executing the current one. Two conditions:
  the new chunk must arrive before the queue empties (`L ≤ e·T`), and its first `ceil(L/T)` actions
  are already stale when it arrives, so the chunk must be long enough (`ceil(L/T) + e ≤ H`).
  Otherwise the robot **starves**: it holds still or replays old actions.

## Today's build (`w12d4_latency.py`)

`nearest_rank`, `latency_stats`, `measure` (with an injected clock), `deadline_miss_rate`,
`sync_idle_fraction`, `stale_steps`, `max_async_latency`, `starvation_steps`.

## Staff angle

"On the Orin, SmolVLA's p95 is 180 ms; at 30 Hz with 50-action chunks executed 25 at a time, async
inference never starves up to 830 ms, sync would idle 18% of the time" is the deployment sentence a
research team rarely has. It also decides whether your eval of a slow policy measured the policy or
its stutter.
