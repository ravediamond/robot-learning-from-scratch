# Predict BEFORE running `predict_run.py`

1. Control at 30 Hz, physics timestep 2 ms. `int((1/30) / 0.002)` →      `round((1/30) / 0.002)` →
2. Arm held horizontal (shoulder_lift = 1.2 rad), kp = 20, kd = 1, no gravity compensation. Steady-state error at that joint: (≈ 0 / a few hundredths of a rad / more than 0.1 rad)? →
3. Same with gravity compensation →
4. kp = 20 → 5 without compensation: the error is multiplied by about →
5. Shape of the end-effector position Jacobian for this arm →        Which joint has a zero column at the zero pose for the ee position? →
6. IK from the zero pose to a reachable target, λ = 0.05: iterations to get under 1 mm: (≈ 5 / ≈ 30 / ≈ 300)? →

## Misses
-
