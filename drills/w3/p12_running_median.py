class RunningMedian:
    """Median of a stream of latency measurements. add(x) O(log n), median() O(1). Two heaps.
    Even count: mean of the two middle values."""

    def add(self, x: float) -> None:
        raise NotImplementedError

    def median(self) -> float:
        raise NotImplementedError
