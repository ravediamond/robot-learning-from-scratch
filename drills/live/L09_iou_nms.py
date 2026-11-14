"""L09 (30 min). Box IoU and non-maximum suppression, vectorised. Boxes are (x1, y1, x2, y2) with x2 > x1, y2 > y1."""
import torch


def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a (N, 4), b (M, 4) -> (N, M) IoU. No Python loops."""
    # TODO
    raise NotImplementedError


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float) -> list[int]:
    """Greedy NMS: repeatedly keep the highest-scoring remaining box and drop the boxes whose IoU with it is
    > iou_thresh. Return kept indices in order of decreasing score (ties: lower index first)."""
    # TODO
    raise NotImplementedError
