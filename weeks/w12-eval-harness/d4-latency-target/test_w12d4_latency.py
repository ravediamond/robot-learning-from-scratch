from w12d4_latency import (deadline_miss_rate, latency_stats, max_async_latency, measure, nearest_rank,
                           starvation_steps, stale_steps, sync_idle_fraction)


def test_nearest_rank_and_stats():
    xs = sorted([10.0] * 95 + [200.0] * 5)
    assert nearest_rank(xs, 0.5) == 10 and nearest_rank(xs, 0.95) == 10 and nearest_rank(xs, 0.99) == 200
    assert nearest_rank([3.0], 0.99) == 3.0
    st = latency_stats([2400.0] + [20.0] * 99, warmup=1)
    assert st == {"n": 99, "p50": 20.0, "p95": 20.0, "p99": 20.0, "max": 20.0, "mean": 20.0}
    assert latency_stats([5.0, 1.0, 3.0, 2.0, 4.0])["p50"] == 3.0


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_measure_excludes_warmup():
    clock = FakeClock()
    costs = iter([2.0, 0.5] + [0.010, 0.012, 0.030])

    def fn():
        clock.t += next(costs)

    out = measure(fn, n=3, warmup=2, clock=clock)
    assert [round(x, 6) for x in out] == [10.0, 12.0, 30.0]


def test_deadline_and_sync():
    assert deadline_miss_rate([10, 33.3, 34, 50], 33.3) == 0.5
    assert abs(sync_idle_fraction(180, 25, 30) - 180 / (180 + 25000 / 30)) < 1e-12


def test_async_bounds():
    assert stale_steps(0, 30) == 0 and stale_steps(33.3, 30) == 1 and stale_steps(100, 30) == 3
    assert stale_steps(34, 30) == 2
    assert abs(max_async_latency(50, 25, 30) - 25000 / 30) < 1e-9
    assert abs(max_async_latency(30, 25, 30) - 5000 / 30) < 1e-9
    assert max_async_latency(20, 25, 30) == 0.0


def test_starvation():
    period = 1000 / 30
    lat = [100, 900, 834, 1000, 300]
    window = 25 * period                   # 833.3 ms
    assert starvation_steps(lat, 25, 30) == 2 + 1 + 5
    assert starvation_steps([window], 25, 30) == 0
