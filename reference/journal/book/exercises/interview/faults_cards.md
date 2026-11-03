# Faults deck: presented as → lived at (16.1c, 8.7, the /faults/ ledger)

Sorting key, the eight layers bottom up (WS-044): **hardware** (arm, servos, cables, cameras, mounts) · **platform** (Jetson image, drivers, builds, thermal) · **engine** (vla.cpp, ggml, the port) · **loader** (checkpoints, key maps, strict load) · **data** (datasets, features, splits, normalisation, leakage) · **model** (architecture, objective, training) · **evaluation** (benchmarks, judges, statistics, bars) · **register** (the log itself: parsing, units, cells).

Drill: read the front, say the layer it lived at, the mechanism and the probe, then turn the card. Source tags: `16.1c` = a row of the table; `8.7` = named in 8.7's connector answer (all four are also 16.1c rows); `ledger` = on /faults/ only.

## platform

**Front:** "LeRobot won't run on the Orin." `16.1c` `8.7`
**Back:** *platform.* No channel ships a cp312 CUDA torch wheel for Jetson; five things built from source (PyTorch, torchcodec, torchvision, FFmpeg, dav1d), six traps, 5 h 33 min of which PyTorch is 92 %. Probe: building it. Changed: the image, merged as LeRobot #4599. (Post 2026-09-09.)

## engine

**Front:** A jump at every chunk boundary in vla.cpp. `16.1c`
**Back:** *engine.* Each chunk starts from fresh noise; ggml lacks backward rules for flash attention and concat, so RTC's guidance is out. Probe: the jump measured at 254 boundaries per delay. Changed: gradient-free prefix overwrite, jump cut 71–73 % at +0.4 ms. (Post 2026-10-25; G13b in vla.cpp.)

## loader

**Front:** A world-model result: G11-A. `16.1c` `8.7`
**Back:** *loader.* `strict=False` plus a key remap that did not fit transformers 5.5.4 left the ViT encoder at random init. Probe: the Jetson strict-load check, 192 missing / 192 unexpected keys. Changed: gate voided (the 12.78 ms timing stands); strict load before any number. (G11-A re-check, 2026-10-18.)

## data

**Front:** "TurboVLA ignores the instruction" — first blamed on our port. `16.1c`
**Back:** *data* (presented at the engine). The instruction is redundant on ~94 % of the frames; the pose leaks the side. Probe: path A, the authors' own trainer on our data, 0.020°, ≈ $2. Changed: port claim withdrawn, data claim stands. (TurboVLA correction, 2026-10-23.)

**Front:** A failure monitor that works: REAP v1, AUROC 0.93. `16.1c`
**Back:** *data.* A time feature plus episode length carried the outcome (elapsed time alone caught 0/8). Probe: REAP v2's controls — no time feature, common 374-frame horizon, shuffled and zero actions. Changed: 0.93 → 0.79, verdict to fail. (REAP v2, 2026-10-23.)

**Front:** C1: a latent probe that might predict failures. `ledger`
**Back:** *data.* Failures were concentrated in one policy: policy identity alone, AUROC 0.76, beat every probe (current 0.68, imagined 0.72). Probe: scoring policy identity as a baseline. Changed: not proven, data confounded with policy. (C1 probe, 2026-10-11.)

**Front:** Failure-onset labels for the 25 failed rollouts. `ledger`
**Back:** *data* (presented as evaluation). VLM-marked onsets: median +6 frames on a spot check, but +110 on one episode. Probe: a 5-episode check against the video. Changed: two episodes sent for hand relabelling; REAP v2 dropped onsets for a ramp target. (Failure onsets, 2026-10-23.)

## evaluation

**Front:** REKI's latency on the Orin, 318 ms (G12). `16.1c` `8.7`
**Back:** *evaluation.* The benchmark timed a random-init Qwen2.5-1.5B without vision tower or Gated DeltaNet plus a hand-written expert. Probe: its parts did not sum, 135.4 + 230.3 ≠ 318.2 ms. Changed: G12 void; the parts-must-sum rule. (G12 re-check, 2026-10-18.)

**Front:** W1's live reflex thresholds 2.2 / 10.0, ready for the arm. `ledger`
**Back:** *evaluation* (the bar). Fit on seen rollouts; on the 50 judge episodes they flag 42/42 successes. Probe: screening on recordings before the robot day (W1-A). Changed: refit (G11-B-cal: 3/25 at 1/44, fail); reflex in shadow mode. (2026-10-19.)

## register

**Front:** One failed gate too many on the scoreboard. `16.1c` `8.7`
**Back:** *register.* Unescaped `|Δ|` pipes in the vla.cpp determinism row shifted its cells and faked a fail. Probe: a hand count of fails against the parser. Changed: pipes escaped; split on unescaped pipes only; 73 gates, 20 failed. (WS-039 log, 2026-10-26.)

**Front:** TurboVLA's LIBERO control "0.105°" on a card. `ledger`
**Back:** *register.* The value is in normalised LIBERO action units, ≈ 5.5° of pan. Probe: an outside audit of the repo's units. Changed: card restated. (WS-033 log, 2026-10-26.)

**Front:** G-flora-abl: one verdict for the ablation. `ledger`
**Back:** *register.* Three arms, three verdicts: 27B pass, 9B fail, Gemini Flash narrow fail. Probe: syncing every gate into the public log. Changed: row restated. (WS-039 log; G-flora-abl, 2026-10-20.)

## hardware

No register row yet. The candidate is the robot day's packet loss on motors 4 and 5 (wrist cabling; retried in session, 2026-10-27); it becomes a card when it becomes a row.
