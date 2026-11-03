from collections import Counter


def top_k_frequent(labels, k):
    return [w for w, _ in sorted(Counter(labels).items(), key=lambda t: (-t[1], t[0]))[:k]]
