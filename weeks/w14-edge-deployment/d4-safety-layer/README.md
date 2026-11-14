# W14 D4: the safety layer between policy and motors

Primer: `knowledge/15-edge-deployment.md` §4.

## The one idea

A learned policy will, eventually, output something wrong: a NaN after a bad frame, a jump to the other
side of the workspace, a command into a joint limit, or a perfectly fine action computed from a camera
frame that is 800 ms old because the USB hub hiccupped. The safety layer is small, boring, deterministic
code that sits between the policy and the motors and that you can test exhaustively:

1. **NaN/inf guard**: hold the last safe command.
2. **Joint limits**: clamp into the calibrated range (minus a margin).
3. **Rate limit**: no joint moves more than `max_step` per control step.
4. **Watchdog**: if the observation's **capture** time is older than `max_age`, hold. Use the camera's
   capture timestamp, not the time the message arrived: a frame can arrive fresh-looking after sitting
   in a buffer.
5. **E-stop latch**: once triggered, stay stopped until an explicit reset, even if the trigger clears.

## Today's build (`w14d4_safety.py`)

`clamp_limits`, `rate_limit`, `is_stale`, `SafetyFilter`.

## Staff angle

In a robotics interview, "what stops your policy from hurting someone?" deserves an answer with layers:
mechanical (torque limits in servo firmware), control (this filter), monitoring (failure detectors from
W6 D3), process (operator with an e-stop, speed limits near humans). Owning this layer is part of
owning deployment.
