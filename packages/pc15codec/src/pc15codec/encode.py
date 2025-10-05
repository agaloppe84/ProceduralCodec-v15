from __future__ import annotations
from dataclasses import dataclass
import torch
from pc15proc.params import ParamCodec
from pc15proc.registry import get as get_gen
from .search import SearchCfg, score_batch
from .tiling import TileGridCfg, blend, tile_image

@dataclass
class EncodeStats:
    tiles: int
    timings_ms: dict
    vram_mb: float
    generators_hist: dict

@dataclass
class EncodeResult:
    bitstream: bytes
    bpp: float
    stats: EncodeStats
    tile_map: list[dict]

def encode_y(img_y: torch.Tensor, grid: TileGridCfg, search: SearchCfg, models: dict) -> EncodeResult:
    device = img_y.device
    tiles = tile_image(img_y, grid)
    out_synth = []
    tile_map = []
    for i in range(tiles.count):
        gen = get_gen("STRIPES")
        pc = ParamCodec(gen.info)
        params = pc.to_tensor({"freq": 6.0, "angle_deg": 0.0, "phase": 0.0}, device=device, dtype=img_y.dtype).unsqueeze(0)
        seeds = torch.tensor([0], device=device, dtype=torch.int64)
        synth = gen.render((grid.size, grid.size), params, seeds, device=device, dtype=img_y.dtype)
        out_synth.append(synth[0])
        tile_map.append({"tile_id": i, "gen": "STRIPES", "qv_id": 0, "seed": 0, "flags": 0})
    tiles_tensor = torch.stack(out_synth, dim=0)
    _ = blend(tiles_tensor, tiles, img_y.shape[-2], img_y.shape[-1])
    stats = EncodeStats(tiles=tiles.count, timings_ms={}, vram_mb=0.0, generators_hist={"STRIPES": tiles.count})
    return EncodeResult(bitstream=b"", bpp=0.0, stats=stats, tile_map=tile_map)
