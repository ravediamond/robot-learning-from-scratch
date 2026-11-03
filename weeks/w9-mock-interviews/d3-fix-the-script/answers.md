# Model answers

1. Personal: write which you missed. Most people miss the data ones (split, stats) and the save/load one, because the script "works".
2. (1)(2) val looks better than reality, wrong model selection; (3) policy sloppy at the end of tasks (place/release); (4) training effectively at a tiny lr for most of the run: underfit; (5) noisy val, random checkpoint choice; (6) the deployed policy outputs actions in normalised units: tiny or wild motions on the robot; (7) momentum-like accumulation: unstable or diverging training.
3. Split leak (1) and stats leak (2) make val look good; save/load stats (6) only shows at deployment. (5) also hides as "noise".
4. Most expensive late: (6) (robot days wasted, looks like a bad policy), then (1)/(2) (wrong conclusions in every comparison), then (3), (4), (7) (visible in curves), (5).
5. Overfit one batch; check the split unit and where stats are computed; read the loss masking; check the schedule plot; eval mode + no_grad for validation; save → reload → identical outputs; a closed-loop eval, not only val loss; seeds and determinism; log the config and hashes.
