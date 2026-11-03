class LRUCache:
    """Cache of decoded video frames. get(key) -> value or None; put(key, value). Both O(1).
    When full, evict the least recently USED (get or put counts as use)."""

    def __init__(self, capacity: int):
        raise NotImplementedError

    def get(self, key):
        raise NotImplementedError

    def put(self, key, value) -> None:
        raise NotImplementedError
