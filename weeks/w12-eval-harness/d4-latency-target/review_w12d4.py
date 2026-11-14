"""REVIEW. Latency report of the deployment dashboard. It says p99 = 117 ms for a policy whose operators
complain about a visible 200 ms freeze every few seconds. Find the bug, fix, replace:
BUG:
"""
import statistics


def p99(samples_ms: list[float]) -> float:
    return statistics.mean(samples_ms) + 2.33 * statistics.pstdev(samples_ms)
