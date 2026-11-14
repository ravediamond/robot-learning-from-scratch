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

import toy_flow
import w14d3_rtc as m

prev = toy_flow.modes(0.0)[0]
print("1. prev[5:], of which the first 3 will execute during inference:", m.shift_previous(prev, 5)[:4].tolist(), "...")
g = torch.Generator().manual_seed(0)
remaining = m.shift_previous(prev, 5)
start = remaining[0].item()
mask = m.soft_mask(toy_flow.H, fixed=3, overlap=8)
naive = [toy_flow.sample(start, 20, g) for _ in range(200)]
rtc = [m.guided_sample(lambda x, t: toy_flow.velocity(x, t, start), remaining, mask, 20, 30.0, g) for _ in range(200)]
down = lambda chunks: sum(1 for c in chunks if c[-1] < start) / len(chunks)
print(f"2. naive: {down(naive):.0%} of new chunks switch mode")
print(f"3. RTC:   {down(rtc):.0%}")
jn = sum(m.seam_jump(remaining, c, 3) for c in naive) / 200
jr = sum(m.seam_jump(remaining, c, 3) for c in rtc) / 200
print(f"4. mean seam jump naive {jn:.3f} vs RTC {jr:.3f}")
