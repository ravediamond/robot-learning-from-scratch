"""L09 (30 min). Box IoU and non-maximum suppression, vectorised. Boxes are (x1, y1, x2, y2) with x2 > x1, y2 > y1."""
import torch


def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a (N, 4), b (M, 4) -> (N, M) IoU. No Python loops."""
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter)


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float) -> list[int]:
    """Greedy NMS: repeatedly keep the highest-scoring remaining box and drop the boxes whose IoU with it is
    > iou_thresh. Return kept indices in order of decreasing score (ties: lower index first)."""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i].item(), i))
    iou = box_iou(boxes, boxes)
    keep, removed = [], set()
    for i in order:
        if i in removed:
            continue
        keep.append(i)
        for j in order:
            if j not in removed and j != i and iou[i, j] > iou_thresh:
                removed.add(j)
    return keep
