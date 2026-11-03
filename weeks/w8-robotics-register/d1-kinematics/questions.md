# Questions (out loud)

1. Why 4×4 homogeneous transforms instead of rotation + translation separately?
2. What is a singularity? Where are the SO-101's, and why does damping help there?
3. Your policies output joint angles, not end-effector poses. Give one reason for each choice (joint vs EE action space) for a VLA.
4. The review bug: why does a transposed rotation still "look plausible"? How do you test rotation code in 10 seconds?
5. Calibration: the real SO-101's zero pose differs from the model by a few degrees per joint. What does that do at the tip (estimate in mm)?
6. Interviewer: "IK returns a solution but the arm hits itself or the table. What is missing from your IK?"
7. Predict item 5: the out-of-reach residual is larger than the 0.249 m you'd get with the arm stretched toward the target. Why can damped least squares stall there? How would you get the stretched answer?
