"""Run AFTER writing your guesses AND after your tests pass."""
import sys as _sys


def _unfinished_lab_hint(exc_type, exc, tb):
    if exc_type is NotImplementedError:
        print("\nThis predict script runs YOUR lab code, which is not finished yet (a TODO still raises NotImplementedError)."
              "\nMake `uv run pytest . -k \"not review\"` green in this folder first, then run me again.", file=_sys.stderr)
    else:
        _sys.__excepthook__(exc_type, exc, tb)


_sys.excepthook = _unfinished_lab_hint
import w8d3_control as m

for name, (kp, ki) in {"P": (2.0, 0.0), "PI": (2.0, 1.0)}.items():
    x = m.simulate_plant(m.PID(kp, ki, 0.0, 0.05), steps=400, delay=0, disturbance=-0.3)
    print(f"1 {name}: final error {1.0 - x[-1]:.3f}")
for kp in (1.0, 4.0, 8.0):
    x = m.simulate_plant(m.PID(kp, 1.0, 0.0, 0.05), steps=400, delay=5, disturbance=-0.3)
    print(f"2 kp {kp}: max {max(x):.2f}  final {x[-1]:.3f}")
print("3 6 steps at 30 Hz =", round(6 / 30 * 1000), "ms")
for mode in ("naive", "shift", "blend"):
    r = m.run_chunked(mode, delay=6)
    print(f"3/4 {mode:5s} tracking error {r['tracking_err']:.4f}  max jump {r['max_jump']:.4f}")
