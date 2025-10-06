from __future__ import annotations
import os
import torch
import pytest

from pc15codec.tiling import TileGridCfg, tile_image, blend

ALLOW_CPU = os.getenv("PC15_ALLOW_CPU_TESTS", "0") == "1"

def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    assert ALLOW_CPU, "Set PC15_ALLOW_CPU_TESTS=1 for CPU runners"
    return torch.device("cpu")

def test_step1_tiling_blend_shapes():
    dev = _device()
    H, W = 256, 256
    y = torch.zeros((1, 1, H, W), device=dev, dtype=torch.float16 if dev.type == "cuda" else torch.float32)

    grid = TileGridCfg(size=128, overlap=16)
    spec = tile_image(y, grid)
    assert spec.count >= 4 and spec.ny >= 2 and spec.nx >= 2

    # Génère N tuiles synthétiques (rampe simple dépendant de l'index)
    tiles = []
    for i in range(spec.count):
        base = torch.full((1, 1, grid.size, grid.size), float(i) / max(1, spec.count - 1),
                          device=dev, dtype=y.dtype)
        tiles.append(base[0])  # [1,1,h,w] → on empile après
    tiles_t = torch.stack(tiles, dim=0)  # [N,1,h,w]

    rec = blend(tiles_t, spec, H, W)
    assert rec.shape == y.shape
    assert torch.isfinite(rec).all()
    # Les valeurs doivent être dans [0,1] (rampe pondérée)
    assert float(rec.min()) >= -1e-6 and float(rec.max()) <= 1.0 + 1e-6
