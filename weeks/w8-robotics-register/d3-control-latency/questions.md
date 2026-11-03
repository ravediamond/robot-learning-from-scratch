# Questions (out loud)

1. What does each PID term do? Why does the SO-101's servo already contain a P(I)D loop, and what does your policy command, then?
2. Why does delay make high gains oscillate? Relate it to your Orin inference latency.
3. Your three chunk modes: numbers, and which one RTC improves on and how (it does more than blending).
4. The review bug: why does the behaviour depend on the control rate? What is the general lesson about units in control code?
5. Two clocks: control at 30 Hz, policy at ~3 Hz. Draw the timeline of one chunk from observation to the last executed action.
6. Interviewer: "Your VLA takes 300 ms on the Orin. Design the execution layer." (Use today's modes and real-time chunking.)
