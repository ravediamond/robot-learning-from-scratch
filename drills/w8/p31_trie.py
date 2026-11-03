class Trie:
    """Instruction autocomplete. insert(word), search(word) -> bool (exact), starts_with(prefix) -> bool."""

    def __init__(self):
        raise NotImplementedError

    def insert(self, word: str) -> None:
        raise NotImplementedError

    def search(self, word: str) -> bool:
        raise NotImplementedError

    def starts_with(self, prefix: str) -> bool:
        raise NotImplementedError
