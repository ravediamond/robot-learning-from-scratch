# 02. Vision-language models (VLMs)

Why it matters: every modern VLA (SmolVLA, π0, OpenVLA, GR00T, REKI) is a VLM with an
action head bolted on. "What does the VLM bring, and what does it cost?" is a standard
question. Book depth: 11a.1 (SmolVLA), 11a.4 (REKI).

## Mechanism

### Step 1: align images and text (CLIP / SigLIP)
Two encoders, one for images, one for text. Train on (image, caption) pairs so matching
pairs have high cosine similarity and non-matching pairs low.

**CLIP (InfoNCE, softmax).** Batch of N pairs, similarity matrix S = (zᵢ · tⱼ) / τ, shape (N, N).
Loss = cross-entropy where the "right class" for row i is column i, both directions:
```
L = ½ [ CE(S, arange(N)) + CE(Sᵀ, arange(N)) ]
```
Every other caption in the batch is a negative: big batches matter (CLIP used 32k).

**SigLIP (sigmoid).** Each (i, j) cell is an independent binary problem: positive on the
diagonal, negative elsewhere: `−log σ(z·(S_ij) )` with z = ±1. No softmax over the batch,
so no global normalization: cheaper to scale, works with smaller batches. SigLIP encoders
are the vision tower of many open VLMs (PaliGemma in π0, SmolVLM in SmolVLA).

What contrastive training gives: **semantic** features ("a red cup") robust to lighting
and viewpoint. What it does NOT give: precise geometry, depth, contact, small offsets.
The pooled embedding throws spatial detail away. This is the core tension for robotics.

### Step 2: plug image tokens into a language model (LLaVA recipe)
```
image -> ViT -> patch features (P, d_v) -> projector (MLP) -> (P', d_lm) "visual tokens"
[visual tokens] + [text tokens] -> decoder LLM -> text
```
- **Projector**: a small MLP mapping vision space into the LLM's embedding space.
- **Token reduction**: P is large (576+ per image). Pixel shuffle (space-to-depth: merge
  each 2×2 or 4×4 block of patch features into one token with more channels) cuts the count
  by 4–16×. SmolVLA uses this to keep **64 visual tokens per frame**.
- **Training stages**: (1) train the projector only, on caption data, encoders frozen;
  (2) unfreeze the LLM, instruction-tune on mixed image-text tasks.

### Step 3: from VLM to VLA (preview of 03)
Keep the VLM as the backbone. Either (a) emit actions as text tokens (RT-2, OpenVLA:
discretize each action dim into 256 bins), or (b) attach an **action expert** that reads
the VLM's hidden states and outputs continuous actions (π0, SmolVLA: flow matching).
SmolVLA also **skips the upper half of the VLM layers** (takes features from a middle layer)
because they are slow and mostly language-specialized.

## What a VLM brings to a robot, honestly

| Brings | Does not bring |
|---|---|
| Language grounding: "the pink bowl", "put it left" | Millimetre geometry, depth, contact |
| Robustness to distractors and lighting (semantic features) | Temporal dynamics (single frames) |
| Pretraining on billions of pairs: fewer demos needed for new objects | Speed: 100s of ms per call on edge hardware |
| A common token interface for many cameras and the instruction | Calibration to your camera / your arm |

Your own evidence to cite: TurboVLA initially "ignored the instruction" because its fusion
from Grounding DINO was never initialized in your port (book 11a.2, register). Language
grounding is the first thing to break, and the hardest to see in val loss.

## Questions

1. Write the CLIP loss in PyTorch from memory (5 lines).
<details><summary>Answer</summary>

```python
z = F.normalize(img_emb, dim=-1); t = F.normalize(txt_emb, dim=-1)   # (N, d)
logits = z @ t.T / tau                                                # (N, N)
labels = torch.arange(N)
loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
```
τ is usually learned (stored as log). This is week 5's first lab.</details>

2. CLIP vs SigLIP: what changes and why does it matter for training?
<details><summary>Answer</summary>Softmax over the batch (each row is a K-way classification) vs an independent sigmoid per pair. Sigmoid needs no all-gather across devices to normalize, works at smaller batch sizes, and scales cheaper. Also handles the fact that some "negatives" may be true matches less badly.</details>

3. Why are CLIP features not enough for grasping?
<details><summary>Answer</summary>They are trained to match captions, which describe what is in the image, not where exactly or how it is oriented. Global pooling and caption-level supervision discard fine spatial detail. Grasping needs centimeter/millimeter geometry and contact state. That is why policies use patch-level (not pooled) features, wrist cameras, and why dense self-supervised features (DINOv2, V-JEPA 2.1) are used in world models (DINO-WM).</details>

4. What is pixel shuffle in a VLM and what does it trade?
<details><summary>Answer</summary>Rearrange a (H, W, C) grid of patch features into (H/r, W/r, C·r²): each output token concatenates an r×r block. Tokens drop by r² (attention cost by r⁴), information is kept in channels, but the LLM now attends at a coarser spatial resolution. For manipulation, too aggressive a shuffle hurts fine localization.</details>

5. Why does SmolVLA take features from a middle layer of the VLM?
<details><summary>Answer</summary>Speed: skipping the upper layers roughly halves VLM compute. And prior work found intermediate layers carry representations as useful as (or more useful than) the last ones for downstream tasks; top layers specialize in next-token text prediction, which the action expert does not need.</details>

6. Frozen VLM vs fine-tuned VLM for a VLA on 50 SO-101 demos?
<details><summary>Answer</summary>Frozen: cheap, keeps language/semantic generality, cannot adapt features to your camera and objects; good with very little data. Fine-tuned (full or LoRA): adapts to your domain, risk of overfitting and of losing language following ("catastrophic forgetting": the model starts ignoring the instruction). Common middle ground: freeze vision, train the action expert + LoRA on the LM. Your REKI work is the frozen-VLM case: cite its result.</details>

7. How would you test whether your VLA actually uses the instruction?
<details><summary>Answer</summary>Counterfactual instructions on the same scene: same start, swap "pink bowl" for "blue bowl" (or a nonsense instruction); if behavior does not change, it ignores language. Measure attention or gradient attribution to text tokens as a secondary signal. Pre-register the bar (e.g. ≥70% follow the swapped target). This is the check that would have caught the TurboVLA port bug early.</details>

8. Interviewer: "Do we even need language for a single-task robot?"
<details><summary>Answer</summary>For one fixed task, no: a state + image policy (ACT, Diffusion Policy) is smaller, faster, and often as good. Language pays off with many tasks, task specification at runtime, and transfer from pretraining. The VLM backbone may still help through its visual pretraining (robust features) even with a constant instruction. Answer with the use case: Uma-style long-horizon tasks with instructions do need it; a fixed kitting cell may not.</details>
