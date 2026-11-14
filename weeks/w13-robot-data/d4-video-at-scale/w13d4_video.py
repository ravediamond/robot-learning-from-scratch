"""W13 D4: storage and decode budgets, shard assignment, shuffle buffer."""
import math
import random


def raw_bytes_per_hour(width: int, height: int, fps: float, channels: int = 3) -> float:
    """Uncompressed bytes for one hour of one camera (1 byte per channel)."""
    # TODO
    raise NotImplementedError


def compressed_tb(hours: float, cameras: int, mbit_per_s: float) -> float:
    """Terabytes (1e12 bytes) for `hours` of `cameras` streams at mbit_per_s megabits per second each."""
    # TODO
    raise NotImplementedError


def expected_decoded_frames(gop: int) -> float:
    """A random frame uniformly placed in a GOP of `gop` frames needs decoding from the keyframe up to itself:
    on average (gop + 1) / 2 frames."""
    # TODO
    raise NotImplementedError


def decode_cores_needed(frames_per_s: float, decode_fps_per_core: float, decoded_per_sample: float) -> int:
    """Cores to decode frames_per_s training samples when each costs decoded_per_sample decoded frames (ceil)."""
    # TODO
    raise NotImplementedError


def assign_shards(n_shards: int, world_size: int, num_workers: int, rank: int, worker: int) -> list[int]:
    """Shards read by one loader worker of one rank. Global reader id = rank * num_workers + worker;
    reader r reads shards r, r + R, r + 2R, ... where R = world_size * num_workers."""
    # TODO
    raise NotImplementedError


class ShuffleBuffer:
    """Streaming shuffle: keep up to `size` items; once full, each new item replaces a random buffered item,
    which is emitted. rng = random.Random(seed)."""

    def __init__(self, size: int, seed: int):
        self.size = size
        self.buf: list = []
        self.rng = random.Random(seed)

    def push(self, item):
        """Add item. Return the emitted item, or None while the buffer is filling."""
        # TODO
        raise NotImplementedError

    def drain(self) -> list:
        """At the end of the stream: emit the remaining items in random order (rng.shuffle) and empty the buffer."""
        # TODO
        raise NotImplementedError
