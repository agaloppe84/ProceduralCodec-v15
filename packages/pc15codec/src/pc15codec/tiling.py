from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class TileGridCfg:
    size: int = 256
    overlap: int = 32

@dataclass
class TileBatch:
    xy: torch.Tensor   # [N,2] int32 (left, top)
    hw: tuple[int, int]
    count: int

def _positions(length: int, size: int, overlap: int) -> list[int]:
    step = max(1, size - overlap)
    pos: list[int] = []
    p = 0
    while True:
        q = min(p, max(0, length - size))
        if not pos or q != pos[-1]:
            pos.append(q)
        if q + size >= length:
            break
        p = p + step
        if p + size > length:
            p = length - size
    return pos

def tile_image(y: torch.Tensor, cfg: TileGridCfg) -> TileBatch:
    _, _, H, W = y.shape
    xs = _positions(W, cfg.size, cfg.overlap)
    ys = _positions(H, cfg.size, cfg.overlap)
    xy = [(x, y0) for y0 in ys for x in xs]
    xy_t = torch.tensor(xy, dtype=torch.int32, device=y.device)
    return TileBatch(xy=xy_t, hw=(cfg.size, cfg.size), count=xy_t.shape[0])

def _hann2d(h: int, w: int, device, dtype):
    wx = torch.hann_window(w, periodic=False, device=device, dtype=dtype)
    wy = torch.hann_window(h, periodic=False, device=device, dtype=dtype)
    return wy.view(h, 1) * wx.view(1, w)

def blend(tiles: torch.Tensor, batch: TileBatch, H: int, W: int) -> torch.Tensor:
    device, dtype = tiles.device, tiles.dtype
    out = torch.zeros((1, 1, H, W), device=device, dtype=dtype)
    weight = torch.zeros_like(out)
    h, w = tiles.shape[-2:]
    win = _hann2d(h, w, device, dtype).view(1, 1, h, w)
    for i in range(batch.count):
        x0, y0 = batch.xy[i].tolist()
        hs = max(0, min(h, H - y0))
        ws = max(0, min(w, W - x0))
        if hs == 0 or ws == 0:
            continue
        out[..., y0:y0+hs, x0:x0+ws] += tiles[i:i+1, :, :hs, :ws] * win[:, :, :hs, :ws]
        weight[..., y0:y0+hs, x0:x0+ws] += win[:, :, :hs, :ws]
    out = out / torch.clamp(weight, min=1e-8)
    return out
