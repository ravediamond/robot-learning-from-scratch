class RingBuffer:
    """Fixed-size telemetry buffer (e.g. last N joint torques). push(x) O(1); mean() O(1) over the stored
    values; values() -> list oldest first."""

    def __init__(self, size: int):
        raise NotImplementedError

    def push(self, x: float) -> None:
        raise NotImplementedError

    def mean(self) -> float:
        raise NotImplementedError

    def values(self) -> list[float]:
        raise NotImplementedError
