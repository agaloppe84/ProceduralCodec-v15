import os
import torch
from pc15codec.tiling import TileGridCfg, tile_image, blend
from pc15proc.stripes import GEN
from pc15proc.params import ParamCodec

ALLOW_CPU = os.getenv("PC15_ALLOW_CPU_TESTS","0")=="1"

def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    assert ALLOW_CPU, "Set PC15_ALLOW_CPU_TESTS=1 for CPU tests."
    return torch.device("cpu")

def test_tile_and_blend():
    dev = device()
    dtype = torch.float16 if dev.type=="cuda" else torch.float32
    y = torch.zeros((1,1,256,256), device=dev, dtype=dtype)
    grid = TileGridCfg(size=128, overlap=16)
    tb = tile_image(y, grid)
    pc = ParamCodec(GEN.info)
    params = pc.to_tensor({"freq":4.0,"angle_deg":45.0,"phase":0.0}, device=dev, dtype=dtype).unsqueeze(0)
    seeds = torch.tensor([0], device=dev)
    tiles = []
    for _ in range(tb.count):
        tiles.append(GEN.render((grid.size, grid.size), params, seeds, device=dev, dtype=dtype)[0])
    tiles_t = torch.stack(tiles, dim=0)
    rec = blend(tiles_t, tb, 256, 256)
    assert rec.shape == y.shape
