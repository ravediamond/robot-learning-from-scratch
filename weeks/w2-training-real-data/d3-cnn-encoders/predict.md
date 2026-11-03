# Predict BEFORE running `predict_run.py`

1. `nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)` on (1, 3, 96, 96): output shape →
2. `F.unfold(torch.zeros(1, 3, 8, 8), kernel_size=3)` shape →
3. Receptive field of three 3×3 convs, stride 1 →   of three 3×3 convs, stride 2 →
4. `nn.BatchNorm1d(4)` in train mode on a batch of 1 sample: output or error? →
5. A frame from a video decoder has shape (480, 640, 3). After `x.view(3, 480, 640)` is it the same image as `x.permute(2, 0, 1)`? →
6. Spatial softmax of a feature map that is constant everywhere: expected (x, y) →

## Misses
-
