# W14 D3: real-time chunking

Primer: `knowledge/15-edge-deployment.md` §3. Paper: Black et al., "Real-Time Execution of Action Chunking Flow Policies" (2025, in your project files). W8 D3 compared naive / shift / blend switching.

## The one idea

Asynchronous inference (W12 D4) keeps the arm moving while the next chunk is generated. Two problems
remain at the seam where the new chunk takes over:

1. The new chunk was computed from an observation that is `d` steps old. Its first `d` actions are
   already in the past: the robot executed the OLD chunk's actions meanwhile.
2. A multimodal policy (flow matching) may pick a different mode for the new chunk (go left of the
   obstacle instead of right): the arm jumps.

RTC treats the new chunk as **inpainting**: the actions that the robot will execute during inference are
known (the previous chunk, **shifted** by the steps already executed), so generate the new chunk
**conditioned** on them. (The paper's guidance is a principled ΠGDM-style correction; today's version, a
masked pull of the denoised estimate towards the known prefix with a fixed strength, shows the same effect.) A soft mask weighs the overlap: weight 1 on the first `d` actions (fixed),
decaying over the rest of the overlap, 0 after. At every flow step, the denoised estimate is pulled
towards the known prefix on masked positions (guidance), and the rest of the chunk follows coherently.

## Today's build (`w14d3_rtc.py`)

`shift_previous`, `soft_mask`, `guided_sample`, `seam_jump`. `toy_flow.py` (provided) is an exact
velocity field for a two-mode chunk distribution (go up or down), so no training is needed.

## Staff angle

RTC is the difference between a VLA that looks jerky in a demo and one that looks smooth at 200 ms
latency. Being able to say why (stale prefix + mode switching) and how (inpainting with a soft mask) is
a strong signal for a role that owns deployment.
