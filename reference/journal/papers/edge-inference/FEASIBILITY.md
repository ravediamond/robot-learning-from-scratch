# Feasibility — "A sync-mode inference recipe for a VLA on an edge GPU": what survives on an Orin NX

WS-071, 2026-10-31. Memo only: nothing was run. Sources: `journal/EXPERIMENTS.md` rows quoted by name,
`shared/gate/data/` READMEs, `projects/{horizon-depth,pact-vla,aac-chunking,smooth-chunking,vlacpp}`, the posts
`smolvla-4x-faster-on-jetson-with-vlacpp`, `twelve-vlas-orin`, `vla-chunk-boundary-jumps`, the papers mirror.

## 1. Claim in one sentence

On a Jetson Orin NX, a 450M SmolVLA spends 151 of 219 ms per chunk in vision (G-lat-2), so of the training-free
latency and smoothness tricks in the literature, only those that touch vision or the seam transfer: a gradient-free
prefix overwrite (−72 % boundary jump at +0.4 ms, G13b / vla.cpp) and a ProMP projection with via-point conditioning
(−83 to −87 % seam jerk, WS-064 A) transfer; fewer Euler steps save 41 ms (−17 %, not the paper's order of
magnitude), and N-sample methods do not fit (PACT 1,200 ms, AAC N = 20 4,751 ms against a 238 ms chunk). True if the
arm gates of §3 show the transferring tricks non-inferior in success.

## 2. Evidence in hand

| gate | date | n | result | status | anchor |
|---|---|---|---|---|---|
| SmolVLA PyTorch / vla.cpp / vla.cpp `_fa` on the arm | 09-29 / 10-01 | 10 trials each | 9/10 at 974 ms · 9/10 at 390 ms · 8/10 at 238 ms | works | EXPERIMENTS rows 23–25; `2026-10-01_orin_nx_vlacpp` |
| G-lat-2 (server split) | 10-20 | 400 chunks | total 219.0 ms: vision 151.0, prefill + denoise 67.3 | pass (its own bar) | `2026-10-20_glat2` |
| G13b offline (RTC vs prefix conditioning) | 10-19 | 20 episodes, 254 boundaries | overwrite −72 to −74 % median jump; exact RTC −21 to −41 %; PyTorch 1,028 ms naive, 1,962 ms exact RTC (1.9×) | propose 1+2 | `2026-10-19_g13b_offline` |
| G13b in vla.cpp | 10-19 | same | −71.0 / 73.3 / 71.5 % (d = 8/12/20), +0.4 ms per chunk, prefix error 0.0 | pass | `2026-10-19_vlacpp_prefix_overwrite` |
| vla.cpp parity / determinism | 10-22 / 10-23 | 8 frames / 1,534 anchors | 0.80° median max-Δ vs PyTorch (noise spread 12.2°); max Δ 0.0° on reload | pass / deterministic | EXPERIMENTS |
| Horizon × depth A1 (N = 2 cloud) | 10-29 | 20 frames × 32 draws | R(2)/R(10) 0.78 (CI 0.73–0.81) | not proven | `projects/horizon-depth` |
| Horizon × depth A2 (latency vs N) | 10-30 | 50 calls per N, MAXN | 196.1 vs 237.5 ms (0.83); 5.3 ms per denoise step | pass | same |
| PACT-VLA gate B / gate A | 10-29 / 10-30 | 254 boundaries / MAXN | selection −0.27° (pass); cycle 1,200 ms vs 237 ms, 37 ticks vs 8 | B pass, A fail | `projects/pact-vla` |
| AAC gate 0 / gate 1 | 10-29 / 10-30 | 254 decisions / N = 1–40 | h* at the floor (219 / 254); N = 20 4,751 ms vs 298 ms bar | fail / fail | `projects/aac-chunking` |
| Smooth MP chunking A1 / A2 | 10-31 | 20 episodes | projection −0.02° error, jerk × 0.06; seam × 0.13–0.17 at +0.19° | pass / pass (arm gate B next) | `projects/smooth-chunking` |
| Twelve VLAs on the Orin | 10-06 | 12 models | latency and memory per model in vla.cpp | measured | post `benchmarking-twelve-vlas-on-an-orin-nx` |

Pre-registered: horizon × depth, PACT, AAC, smooth chunking (`PROTOCOL.md` before any number); G13b (bar in
`projects/vlacpp/README.md` §7 before the run); G-lat-2 (`82fdc15`). The 10-trial arm numbers of 09-29 / 10-01 predate
the bar discipline. Statistics: episode bootstrap on the offline gates; latencies are medians over ≥ 50 calls.
WS-067 reran horizon-depth A1, AAC gate 0 and PACT gate B cold: identical.

## 3. Gaps, costed

| gap | resource | queued? |
|---|---|---|
| **the arm column**: success with each transferring trick, 20 + 20 trials sync (overwrite needs the robot client to send `prev_actions`) | ~1 arm day per trick × 3 (overwrite, ProMP + cond, N = 2) | smooth chunking gate B (WS-064, `go arm`); horizon × depth gate B; overwrite new |
| vision-prefix reuse (LOWN-inspired: skip re-encoding unchanged frames) — the only trick aimed at the 151 ms | ~2 Orin days to build + measure, $0 | new (LOWN 2610.36671 not in the mirror) |
| a second policy (ACT via vla.cpp, already ported) through the same table | ~1 Orin day offline + 1 arm day | new |
| MAXN vs 25 W for every latency row (thermal confound) | ~2 h at MAXN, owner sudo | partly in horizon × depth A4 |
| FLASHRT-GAPS (idle-gap / kernel share in one chunk) | ~0.5 Orin day | frozen 2026-10-23, parked (WS-017) |
| **total** | **≈ 4–5 arm days, ~4 Orin days, $0** | |

## 4. Related work

| paper | what it showed | what differs here |
|---|---|---|
| 2506.07339 Real-Time Chunking | inpainting the committed prefix keeps throughput flat under latency (π0.5, RTX 4090) | exact RTC costs 1.9× on the Orin; the gradient-free overwrite gets most of the jump cut for +0.4 ms |
| 2610.39451 Smooth MP chunking | ProMP projection and via-point conditioning remove jitter without retraining | reproduced offline on SmolVLA (seam −83 to −87 %); arm gate pending |
| 2607.27205 TurboVLA | 32 Hz on an RTX 4090 without a VLM backbone | register: withdrawn here; desktop-GPU number |
| 2610.05719 RACE (when to switch) | predict subskill transitions to extend chunks safely | a learned switch; AAC's training-free switch failed here |
| 2606.18589 DREAM-Chunk | latent world model selects chunks at test time | an N-candidate method: the N-sample cost (AAC, PACT) is the obstacle on the Orin |
| 2608.23831 ARLI (act while waiting) | latency-aware RL fine-tuning | training-based; ours are training-free |
| 2607.01804 VLA-Corrector | event-triggered correction of the action horizon | its monitor failed here (W1-A drift 0/8) |
| horizon × depth 2610.32102, PACT-VLA 2610.28019, AAC 2604.04161, LOWN 2610.36671 | — | **not read here** (not in the mirror) |

## 5. Objections

**First: this is engineering, not science — where is the claim?** The claim is the transfer failures, each a
pre-registered gate with its bar: PACT's "no added control-step delay" (37 vs 8 ticks), AAC's entropy-chosen length
(at the floor 86 % of the time), N = 2's speed-up (−17 %, because vision dominates). The recipe is the residue.

**Second: one policy, one engine written by the author.** vla.cpp is measured against PyTorch on the same box (parity
0.80°, determinism 0.0°); π0-class models do not fit the Orin at all, which is itself the point of the condition
line. A second policy (ACT, already ported) is in §3. Thermal and power-mode confounds: every latency row states 25 W
or MAXN; A4 measured both.

## 6. Venue

Primary: a 2027 workshop on efficient / embodied deployment (RSS or CoRL, deadlines to confirm). Fallback: RA-L
(rolling). arXiv first: yes, after the arm column for at least the overwrite and ProMP rows. Endorsement (cs.RO): no
contacts list in the repo; none named.

## 7. Minimal vs full paper

4 pages: the latency split, the table (trick × ms × offline effect × transfers?), the three transfer failures, the
recipe. Needs: nothing new (all rows exist), arm column marked `[to run]`. 8 pages: + the arm column (3 tricks × 20 +
20), vision-prefix reuse, a second policy, MAXN vs 25 W.

## 8. Risk of a null result

If the transferring tricks are not non-inferior on the arm, the paper becomes "nothing training-free fixes a
vision-bound edge VLA except the seam" — still a useful negative table for anyone deploying on a Jetson.

## 9. Recommendation

**go** for the 4-page version now (every number exists); **go-after-X** for 8 pages, X = the three arm gates
(~3–4 arm days, ~2 weeks at the owner's pace). Earliest: 4-page arXiv late November 2026; 8-page mid-December 2026.

## 10. Skeleton

Titles: (1) *Vision is the bottleneck: training-free inference tricks for a VLA on a Jetson Orin NX*; (2) *What
transfers to the edge: a sync-mode recipe for SmolVLA on an Orin*; (3) *Fewer steps, smoother seams, no
speculation: VLA inference on a $500 GPU*.

Abstract (≤ 150 words): Training-free tricks promise faster or smoother VLA execution, measured on desktop GPUs. On a
Jetson Orin NX, SmolVLA spends 151 of 219 ms per chunk in vision. We measured seven tricks with one checkpoint and the
same 20 episodes. A gradient-free prefix overwrite cuts the boundary jump by 72 % for 0.4 ms; projecting chunks onto
ten movement primitives with a via-point at the seam removes 83–87 % of seam jerk at +0.19°. Two Euler steps save
41 ms (17 %). Exact real-time chunking costs 1.9× a chunk; speculative selection (PACT) 1,200 ms per cycle, 37 control
ticks instead of 8; entropy-adaptive chunking 4.7 s per decision. On the arm, [to run] success for the three that
transfer. The recipe: overwrite the prefix, smooth the seam, leave the sampler alone.

Sections: 1 Intro · 2 Where the time goes (G-lat-2 split) · 3 Seven tricks, one table · 4 Three that do not transfer
and why · 5 On the arm `[to run]` · 6 Recipe · 7 Limitations.

Figures: (1) latency split per chunk — `2026-10-20_glat2` (today); (2) ms × offline effect scatter, one point per trick
— the rows above (today); (3) boundary jump distributions naive / overwrite / exact — `2026-10-19_g13b_offline`
(today); (4) arm success per trick — `[to run]`.
