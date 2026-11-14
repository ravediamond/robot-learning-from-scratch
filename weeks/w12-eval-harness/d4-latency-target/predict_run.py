"""Run AFTER writing your guesses AND after your tests pass."""
import sys as _sys


def _unfinished_lab_hint(exc_type, exc, tb):
    if exc_type is NotImplementedError:
        print("\nThis predict script runs YOUR lab code, which is not finished yet (a TODO still raises NotImplementedError)."
              "\nMake `uv run pytest . -k \"not review\"` green in this folder first, then run me again.", file=_sys.stderr)
    else:
        _sys.__excepthook__(exc_type, exc, tb)


_sys.excepthook = _unfinished_lab_hint
import statistics

import w12d4_latency as m

s = [10.0] * 95 + [200.0] * 5
st = m.latency_stats(s)
print(f"1. p50 {st['p50']} p95 {st['p95']} p99 {st['p99']};  mean + 2.33 sd = {statistics.mean(s) + 2.33 * statistics.pstdev(s):.0f} ms")
print(f"2. sync idle fraction {m.sync_idle_fraction(180, 25, 30):.0%}")
print(f"3. async, H=50: up to {m.max_async_latency(50, 25, 30):.0f} ms")
print(f"4. async, H=30: up to {m.max_async_latency(30, 25, 30):.0f} ms (the stale prefix eats the chunk)")
warm = [2400.0] + [20.0] * 999
a, b = m.latency_stats(warm), m.latency_stats(warm, warmup=1)
print(f"5. mean {a['mean']:.1f} vs {b['mean']:.1f}; max {a['max']} vs {b['max']}; p99 unchanged {a['p99']} == {b['p99']}")
