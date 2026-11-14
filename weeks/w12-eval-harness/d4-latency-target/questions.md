# Questions (out loud)

1. How do you measure inference latency correctly on a Jetson? List every trap.
2. Why report p95/p99 and a deadline-miss rate rather than the mean?
3. Derive the conditions for asynchronous chunked execution not to starve. What do you change when they fail?
4. Sync vs async chunking: what does each do to the behaviour you evaluate, beyond speed?
5. What is real-time chunking (RTC), and which of today's failure modes does it address?
6. A policy is 2× slower on the Orin than on your laptop GPU. Give five things to check before blaming the model.
