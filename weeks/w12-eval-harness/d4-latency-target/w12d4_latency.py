"""W12 D4: latency statistics on the target, and synchronous vs asynchronous chunk execution."""
import math


def nearest_rank(sorted_vals: list[float], q: float) -> float:
    """q in (0, 1]: the value at position ceil(q * n) - 1 of the sorted list."""
    # TODO
    raise NotImplementedError


def latency_stats(samples_ms: list[float], warmup: int = 0) -> dict:
    """Drop the first `warmup` samples, then {"n", "p50", "p95", "p99", "max", "mean"} (nearest rank)."""
    # TODO
    raise NotImplementedError


def measure(fn, n: int, warmup: int, clock) -> list[float]:
    """Call fn() warmup + n times; return the n durations in MILLISECONDS of the calls after warm-up.
    clock() returns seconds (time.perf_counter in real use; on a GPU, synchronise inside fn before it returns)."""
    # TODO
    raise NotImplementedError


def deadline_miss_rate(samples_ms: list[float], budget_ms: float) -> float:
    """Fraction of samples strictly above the budget."""
    # TODO
    raise NotImplementedError


def sync_idle_fraction(infer_ms: float, exec_steps: int, hz: float) -> float:
    """Synchronous chunking: infer (robot idle), then execute exec_steps actions at hz. Idle share of the time."""
    # TODO
    raise NotImplementedError


def stale_steps(infer_ms: float, hz: float) -> int:
    """Control steps that elapse during an inference: ceil(infer_ms / period_ms) (0 for 0 ms)."""
    # TODO
    raise NotImplementedError


def max_async_latency(chunk: int, exec_steps: int, hz: float) -> float:
    """Asynchronous chunking: inference starts when a chunk starts executing. The largest latency L (ms) such that
    (1) the new chunk arrives before the exec_steps actions are used up: L <= exec_steps * period, and
    (2) after skipping its stale prefix, the new chunk still holds exec_steps actions: stale_steps(L) + exec_steps <= chunk,
    i.e. L <= (chunk - exec_steps) * period. Return min of the two bounds (0 if negative)."""
    # TODO
    raise NotImplementedError


def starvation_steps(infer_ms_list: list[float], exec_steps: int, hz: float) -> int:
    """Asynchronous execution with measured latencies: chunk i starts executing and the request for chunk i + 1
    is sent at the same moment; it takes infer_ms_list[i]. If it takes longer than the exec_steps * period window,
    the robot starves for ceil((L - window) / period) control steps before chunk i + 1 starts.
    Return the total starved steps over all requests."""
    # TODO
    raise NotImplementedError
