"""PROVIDED tiny image-caption data. Do not edit.

Image (3, 16, 16): one shape (square or plus) of one colour (red, green, blue) in one quadrant.
Caption: 3 tokens [colour, shape, place] from a 9-word vocabulary:
    0 red 1 green 2 blue | 3 square 4 plus | 5 top-left 6 top-right 7 bottom-left 8 bottom-right
24 distinct captions. Many images share a caption: a CLIP batch contains duplicates.
"""
import torch

WORDS = ["red", "green", "blue", "square", "plus", "top-left", "top-right", "bottom-left", "bottom-right"]


def all_captions() -> torch.Tensor:
    return torch.tensor([[c, 3 + s, 5 + p] for c in range(3) for s in range(2) for p in range(4)])   # (24, 3)


def caption_id(cap: torch.Tensor) -> torch.Tensor:
    return cap[:, 0] * 8 + (cap[:, 1] - 3) * 4 + (cap[:, 2] - 5)


def make_shapes(n: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    color = torch.randint(0, 3, (n,), generator=generator)
    shape = torch.randint(0, 2, (n,), generator=generator)
    place = torch.randint(0, 4, (n,), generator=generator)
    imgs = 0.1 * torch.rand(n, 3, 16, 16, generator=generator)
    for i in range(n):
        r0 = 1 + 8 * (int(place[i]) // 2) + int(torch.randint(0, 2, (1,), generator=generator))
        c0 = 1 + 8 * (int(place[i]) % 2) + int(torch.randint(0, 2, (1,), generator=generator))
        if shape[i] == 0:
            imgs[i, color[i], r0:r0 + 5, c0:c0 + 5] = 1.0
        else:
            imgs[i, color[i], r0 + 2, c0:c0 + 5] = 1.0
            imgs[i, color[i], r0:r0 + 5, c0 + 2] = 1.0
    return imgs, torch.stack([color, 3 + shape, 5 + place], 1)
