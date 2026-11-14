"""Run AFTER writing your guesses AND after your tests pass."""
import sys as _sys


def _unfinished_lab_hint(exc_type, exc, tb):
    if exc_type is NotImplementedError:
        print("\nThis predict script runs YOUR lab code, which is not finished yet (a TODO still raises NotImplementedError)."
              "\nMake `uv run pytest . -k \"not review\"` green in this folder first, then run me again.", file=_sys.stderr)
    else:
        _sys.__excepthook__(exc_type, exc, tb)


_sys.excepthook = _unfinished_lab_hint
import torch

import w12d5_report as m

a = {"w": torch.zeros(10_000)}
b = {"w": torch.zeros(10_000)}
b["w"][5_000] = 1.0
print(f"1. str equal: {str(a) == str(b)};  fingerprints equal: {m.tensor_fingerprint(a) == m.tensor_fingerprint(b)}")
print(f"2. fp32 vs fp16: {m.tensor_fingerprint(a) == m.tensor_fingerprint({'w': a['w'].half()})} (dtype and bytes are part of it)")
x = m.run_manifest("abc123", "ff00", "v3", {"policy": {"chunk": 50}}, {"device": "orin"}, {"torch": "2.8"})
y = m.run_manifest("abc123", "ff00", "v3", {"policy": {"chunk": 25}}, {"device": "orin"}, {"torch": "2.8"})
print("3.", m.manifest_diff(x, y))
print("4.")
print(m.results_table([{"policy": "smolvla", "k": 31, "n": 50, "top_failure": "grasp", "p95_ms": 182.4}]))
