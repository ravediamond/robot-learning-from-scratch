# System design: data flywheel for 10 arms at customer sites

Rubric (score yourself 0–2 on each after the recording): requirements stated · data flow drawn ·
humans in the loop placed · metrics at every stage · failure modes + monitors · safety · cost/latency
numbers · what you would build first and why.

1. Requirements and constraints (task, success definition, latency, compute per arm, connectivity, privacy):
2. On the robot (policy, execution layer, monitors, what is logged, how much per day):
3. Upload and storage (what, when, format, size per arm per day):
4. Curation (which episodes become training data; failure labels; human review budget):
5. Training (frequency, base model, fine-tune recipe, compute):
6. Evaluation gate before redeploy (offline filters, sim/replay, fixed real starts, paired, bars):
7. Rollout (canary arm, rollback rule, monitoring after deploy):
8. Where the humans are (teleop corrections, labelling, approval):
9. The first two weeks: what you build first, what you measure, and what result would change the plan:
