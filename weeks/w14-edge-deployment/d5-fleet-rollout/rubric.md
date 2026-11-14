# Rubric

Score each line 0 / 1 / 2.

| # | Criterion | Senior signal (1) | Staff signal (2) |
|---|---|---|---|
| 1 | Requirements | stations and shifts restated | throughput target, success/intervention SLOs, operator ratio, safety requirements, offline operation, non-goals |
| 2 | Latency budget | "runs on the Orin" | per-stage budget (capture, preprocessing, backbone, flow steps, postprocessing), measured percentiles, precision choices, async execution with the starvation condition checked |
| 3 | Runtime | "a Python script" | separate processes for cameras, policy, control loop and safety; timestamps on one clock; RTC or equivalent at chunk seams; deterministic safety filter; local logging |
| 4 | Safety | "e-stop" | layered: firmware limits, safety filter (limits, rates, watchdog on capture time, NaN guard, latch), monitors, operator e-stop, procedures |
| 5 | Monitoring | dashboards | per-station success, interventions per hour, safety events, latency, hardware health; drift detection per station; buffered telemetry for flaky networks |
| 6 | Releases | "push new model" | artifact fingerprint and parity, shadow mode, canary with a pre-registered rollback test, staged rollout, two versions on device, rollback at task boundaries |
| 7 | Data loop | "collect data" | interventions as corrections, failure mining, uploads in off-hours, QA and ledger, targeted collection on site |
| 8 | Lab vs site | none | site acceptance test with the blocked protocol on site objects; expected drop stated; early detection plan in week one |
| 9 | Fleet config | none | desired-state per station (software, firmware, calibration, model), reconciliation and drift alerts |
| 10 | Judgment | builds everything | v1 = 5 stations, manual operator reports + automatic metrics; what is bought (fleet management, remote access); what changes at 200 stations |

0–9 senior or below · 10–14 senior+ · 15–17 staff · 18–20 strong staff.
