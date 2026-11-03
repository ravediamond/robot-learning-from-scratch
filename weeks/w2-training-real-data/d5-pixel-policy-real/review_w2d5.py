"""REVIEW. SOLUTION.

BUG: training augmentation (random_shift) is applied at evaluation: val is measured on randomly shifted
images, so it is noisy and not the deployment condition. Augment only in training. (Val < train usually
means train is measured with augmentation/dropout and val is not: compare like with like.)
"""
import torch


def evaluate_policy(policy, data, aug_pad: int = 4) -> float:
    policy.eval()
    with torch.no_grad():
        pred = policy(data["img"], data["state"])
        err = ((pred - data["actions"]) ** 2).mean(-1)
        return (err * data["mask"]).sum().item() / data["mask"].sum().item()
