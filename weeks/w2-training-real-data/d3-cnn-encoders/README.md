# W2 D3: CNN encoders for policies

Book: P2.5 (layers), 10a.7 (cameras). Used in ACT, Diffusion Policy, robomimic.

## Convolution in one line
A conv slides a small kernel over the image. It is a matmul in disguise:
`F.unfold` cuts every k×k patch into a column → `(B, C·k·k, L)`; multiply by the kernel
reshaped to `(C_out, C·k·k)`; reshape L back to `(H_out, W_out)`.
`H_out = floor((H + 2p − k) / s) + 1`.

**Receptive field**: how many input pixels one output pixel "sees". With layers (kᵢ, sᵢ):
`r = 1, j = 1; for each layer: r += (k − 1)·j; j *= s`. If the gripper is 40 px wide and your
receptive field is 15 px, no single feature sees the whole gripper.

## Spatial softmax (keypoints)
For each channel, softmax over all H·W positions → a probability map → its expected (x, y)
in [−1, 1]. Output `(B, 2C)`: C learned "keypoints". Keeps WHERE things are, which global
average pooling throws away. Standard in robomimic / Diffusion Policy.

## Normalization for policies
BatchNorm uses batch statistics in training and running averages at test time. At inference
on the arm the batch is 1, and training batches of frames from the same episode are
correlated → train/test mismatch. Policies use **GroupNorm** or LayerNorm (no batch dependence).
Diffusion Policy replaces every BatchNorm in its ResNet with GroupNorm.

## Random shift augmentation (DrQ)
Pad the image by p pixels (replicate the border), crop back to the original size at a random
offset, independently per sample. Cheap, and one of the most effective augmentations for
image-based policies: it teaches small camera-pose invariance.

## Today's build (`w2d3_cnn.py`)
`conv2d_unfold`, `receptive_field`, `SpatialSoftmax`, `CNNEncoder` (GroupNorm + spatial
softmax), `random_shift`.

Run: `uv run pytest weeks/w2-training-real-data/d3-cnn-encoders`
