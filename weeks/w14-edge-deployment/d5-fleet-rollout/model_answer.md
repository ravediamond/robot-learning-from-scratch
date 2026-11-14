# Model answer: VLA on 20 warehouse stations

**Numbers.** 20 stations × 16 h ≈ 320 station-hours/day. Target e.g. 300 picks/station/hour at ≥ 95% success
with fewer than 2 interventions per hour per station (one operator for 5 stations must not be saturated).
Lab 85% → expect 60–80% on day one on unseen items: plan for it.

**Runtime on each Orin.** Processes: camera capture (hardware timestamps), policy server (prefix encoding of
the backbone in TensorRT fp16 once per observation, flow head with 5–10 steps), control loop at 30 Hz with the
action queue and RTC at seams, safety filter, logger. One monotonic clock for all timestamps.

**Latency budget.** Capture + preprocessing 15 ms, backbone 80 ms, flow head 10 × 4 ms, postprocessing 5 ms →
~140 ms p50, measured p95 < 200 ms. Chunks of 50 executed 25 at a time: async never starves up to 833 ms;
RTC freezes ceil(200/33) = 7 actions. Sync is not an option (18%+ idle stutter).

**Safety.** Servo torque limits in firmware; safety filter (limits with margins, rate limits, watchdog on capture
time 150 ms, NaN guard, latched e-stop); failure-detector monitor slows or stops the arm; cell guarding and
operator e-stops; procedures for resets.

**Monitoring.** Per station: picks/hour, success, interventions/hour, safety events, latency percentiles,
temperatures, camera frame drops; telemetry buffered locally and synced when the network allows; dashboards
with control limits per station; alerts on drift (CUSUM on success per station).

**Releases.** Monthly model: evaluation in the lab with the blocked protocol on customer items → export, parity,
fingerprint → shadow on 2 stations for a shift → canary on 2 stations for ≥ 40 station-hours with the rate test on
interventions → 10 → 20. Previous version kept on device; rollback at task boundary, automatic on safety events.

**Data loop.** Interventions logged with executed actions; failures flagged by detectors and operators; uploads
at night; QA, ledger, review queue; monthly targeted collection on the customer's hardest items.

**Site acceptance.** Before go-live: blocked evaluation on site with the customer's item mix; week one:
daily review of interventions per station and per item class.

**v1.** 5 stations, then 20. Buy fleet management and remote access; build the runtime, the safety filter and
the release pipeline.
