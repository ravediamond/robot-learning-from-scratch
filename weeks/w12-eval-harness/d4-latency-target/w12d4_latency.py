"""W12 D4: latency statistics on the target, and synchronous vs asynchronous chunk execution."""
import math


def nearest_rank(sorted_vals: list[float], q: float) -> float:
    """q in (0, 1]: the value at position ceil(q * n) - 1 of the sorted list."""
    return sorted_vals[max(0, math.ceil(q * len(sorted_vals)) - 1)]


def latency_stats(samples_ms: list[float], warmup: int = 0) -> dict:
    """Drop the first `warmup` samples, then {"n", "p50", "p95", "p99", "max", "mean"} (nearest rank)."""
    xs = sorted(samples_ms[warmup:])
    return {"n": len(xs), "p50": nearest_rank(xs, 0.5), "p95": nearest_rank(xs, 0.95), "p99": nearest_rank(xs, 0.99),
            "max": xs[-1], "mean": sum(xs) / len(xs)}


def measure(fn, n: int, warmup: int, clock) -> list[float]:
    """Call fn() warmup + n times; return the n durations in MILLISECONDS of the calls after warm-up.
    clock() returns seconds (time.perf_counter in real use; on a GPU, synchronise inside fn before it returns)."""
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(n):
        t0 = clock()
        fn()
        out.append((clock() - t0) * 1000.0)
    return out


def deadline_miss_rate(samples_ms: list[float], budget_ms: float) -> float:
    """Fraction of samples strictly above the budget."""
    return sum(1 for x in samples_ms if x > budget_ms) / len(samples_ms)


def sync_idle_fraction(infer_ms: float, exec_steps: int, hz: float) -> float:
    """Synchronous chunking: infer (robot idle), then execute exec_steps actions at hz. Idle share of the time."""
    return infer_ms / (infer_ms + exec_steps * 1000.0 / hz)


def stale_steps(infer_ms: float, hz: float) -> int:
    """Control steps that elapse during an inference: ceil(infer_ms / period_ms) (0 for 0 ms)."""
    return math.ceil(infer_ms / (1000.0 / hz) - 1e-9) if infer_ms > 0 else 0


def max_async_latency(chunk: int, exec_steps: int, hz: float) -> float:
    """Asynchronous chunking: inference starts when a chunk starts executing. The largest latency L (ms) such that
    (1) the new chunk arrives before the exec_steps actions are used up: L <= exec_steps * period, and
    (2) after skipping its stale prefix, the new chunk still holds exec_steps actions: stale_steps(L) + exec_steps <= chunk,
    i.e. L <= (chunk - exec_steps) * period. Return min of the two bounds (0 if negative)."""
    period = 1000.0 / hz
    return max(0.0, min(exec_steps * period, (chunk - exec_steps) * period))


def starvation_steps(infer_ms_list: list[float], exec_steps: int, hz: float) -> int:
    """Asynchronous execution with measured latencies: chunk i starts executing and the request for chunk i + 1
    is sent at the same moment; it takes infer_ms_list[i]. If it takes longer than the exec_steps * period window,
    the robot starves for ceil((L - window) / period) control steps before chunk i + 1 starts.
    Return the total starved steps over all requests."""
    period = 1000.0 / hz
    window = exec_steps * period
    return sum(math.ceil((L - window) / period - 1e-9) for L in infer_ms_list if L > window)
