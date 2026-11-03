# W8 D3: Control loops, latency, and chunk boundaries (a small RTC simulator)

Book: 10a.4–10a.5 (P/PD/PID, two clocks), 11.4 (real-time execution, the boundary problem), 11a.5 (RTC).
Coding list item 13. Your edge-inference paper is about exactly this budget.

## PID in one screen
```
e = target − measured
u = kp·e + ki·Σe·dt + kd·(e − e_prev)/dt
```
P alone leaves a steady-state error against a constant disturbance; I removes it (and can overshoot /
wind up); D damps. Delay in the loop (sensing, compute, comms) makes high gains oscillate.

## Chunks with inference delay
A policy plans a chunk of H future commands from an observation at time t. Inference takes d control
steps, so the chunk becomes available at t + d. Three ways to switch to it:
- **naive**: execute `chunk[0]` at t + d: the commands are d steps stale → lag and a jump.
- **shift**: execute `chunk[d]` at t + d (skip what is already in the past). No lag; still a jump if the
  new chunk disagrees with the old one (e.g. a different sample/mode).
- **blend** (RTC idea, simplified): over the first m steps after the switch, cross-fade from the old
  chunk's remaining commands to the new chunk (weights ramp 0 → 1). Smooth boundaries.

## Today's build (`w8d3_control.py`)
`PID`, `simulate_plant` (provided-style loop you complete), `run_chunked(mode)`.

Run: `uv run pytest weeks/w8-robotics-register/d3-control-latency`
