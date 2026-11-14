# Questions (out loud)

1. Write the SAC critic target and say what each of its three ideas (bootstrap, twin minimum, entropy term) is for.
2. Why does the tanh squash need a log-prob correction? What happens to training without it?
3. What is the "deadly triad", and which SAC components reduce its danger?
4. HIL-SERL trains real-robot skills in 1–2 hours with SAC-style RL. Name the ingredients that make this possible.
5. Update-to-data ratio (UTD): what is it, why raise it on a real robot, and what goes wrong when it is too high?
6. Your critic loss goes down while the Q-values grow without bound. Diagnosis and fixes.
