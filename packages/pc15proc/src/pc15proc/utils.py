from __future__ import annotations
import torch
from pc15core.device import get_device

def grid(h: int, w: int, *, device=None, dtype=None):
    if device is None:
        device = get_device(strict_gpu=False)
    if dtype is None:
        dtype = torch.float16
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h, device=device, dtype=dtype),
        torch.linspace(-1, 1, w, device=device, dtype=dtype),
        indexing="ij",
    )
    return xx, yy
