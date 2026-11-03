# Questions (out loud)

1. Spatial softmax vs global average pooling: what does each keep, and which would you use for grasping? Why?
2. Why GroupNorm instead of BatchNorm in Diffusion Policy? Give the inference-time failure in one sentence.
3. Your images are 480×640 and the policy uses 96×96. Resize or crop? What can each destroy?
4. The review bug: describe what the network actually sees. Why can it still learn something?
5. Random shift: why is it so effective for policies, and when would it hurt (think about a fixed wrist camera vs a top camera)?
6. Interviewer: "Pretrained ResNet-18 (ImageNet) or train a small CNN from scratch, for 50 demos?" Decide and justify.
