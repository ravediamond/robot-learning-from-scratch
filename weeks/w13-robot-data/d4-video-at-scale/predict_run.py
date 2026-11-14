"""Run AFTER writing your guesses AND after your tests pass."""
import sys as _sys


def _unfinished_lab_hint(exc_type, exc, tb):
    if exc_type is NotImplementedError:
        print("\nThis predict script runs YOUR lab code, which is not finished yet (a TODO still raises NotImplementedError)."
              "\nMake `uv run pytest . -k \"not review\"` green in this folder first, then run me again.", file=_sys.stderr)
    else:
        _sys.__excepthook__(exc_type, exc, tb)


_sys.excepthook = _unfinished_lab_hint
import w13d4_video as m

gb = m.raw_bytes_per_hour(640, 480, 30) / 1e9
print(f"1. {gb:.0f} GB/hour raw per camera, {3 * gb:.0f} GB/hour for three")
print(f"2. {m.compressed_tb(10_000, 3, 1.0):.1f} TB")
print(f"3. GOP 60: {m.expected_decoded_frames(60):.1f} frames; GOP 15: {m.expected_decoded_frames(15):.1f}")
print(f"4. {m.decode_cores_needed(4000, 400, m.expected_decoded_frames(60))} cores")
a = m.assign_shards(64, world_size=4, num_workers=4, rank=0, worker=0)
print(f"5. {len(a)} shards per (rank, worker): {a}. Ignoring the worker id: every worker of a rank reads the same 16 shards (each sample 4x).")
