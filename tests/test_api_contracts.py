import logging
import os
import torch

from pc15proc.register_all import register_all
from pc15proc.registry import list_generators, get
from pc15proc.params import ParamCodec

from pc15codec.tiling import TileGridCfg, tile_image, blend
from pc15metrics.psnr_ssim import psnr, ssim

log = logging.getLogger("pc15.tests.api_contracts")

ALLOW_CPU = os.getenv("PC15_ALLOW_CPU_TESTS", "0") == "1"

def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    assert ALLOW_CPU, "Set PC15_ALLOW_CPU_TESTS=1 for CPU runners"
    return torch.device("cpu")

def _mid_params(info):
    params = {}
    for p in info.param_specs:
        if p.type in ("float", "int") and p.range is not None:
            lo, hi = p.range
            x = (float(lo) + float(hi)) / 2.0
            if p.type == "int":
                x = int(round(x))
            params[p.name] = x
        elif p.type == "enum" and p.enum is not None:
            params[p.name] = p.enum[0]
        elif p.type == "bool":
            params[p.name] = False
        else:
            params[p.name] = 0.0
    return params

def test_api_contracts_global():
    dev = _device()
    dtype = torch.float16 if dev.type == "cuda" else torch.float32

    # 1) Auto-discovery & registry
    names = register_all(verbose=False)
    infos = list_generators()
    log.info("Discovered %d generators; sample: %s", len(infos), ", ".join(n.info.name for n in infos[:10]))
    assert len(infos) >= 20, "At least 20 generators expected"
    # uniqueness of names
    all_names = [i.name for i in infos]
    assert len(set(all_names)) == len(all_names)

    # 2) Basic contract of a canonical generator (STRIPES)
    g = get("STRIPES")
    assert hasattr(g, "render")
    assert hasattr(g, "info") and hasattr(g.info, "name") and hasattr(g.info, "param_specs")
    pc = ParamCodec(g.info)
    params = _mid_params(g.info)
    P = pc.to_tensor(params, device=dev, dtype=dtype).unsqueeze(0)
    seeds = torch.tensor([0], device=dev, dtype=torch.int64)
    y = g.render((64, 64), P, seeds, device=dev, dtype=dtype)
    log.info("Render STRIPES -> shape=%s dtype=%s stats(min=%.4f max=%.4f mean=%.4f)",
             tuple(y.shape), y.dtype, y.min().item(), y.max().item(), y.mean().item())
    assert y.shape == (1, 1, 64, 64)
    assert torch.isfinite(y).all()

    # 3) Tiling/Blend path (light)
    grid = TileGridCfg(size=32, overlap=8)
    tb = tile_image(y, grid)
    tiles = []
    for _ in range(tb.count):
        tiles.append(g.render((grid.size, grid.size), P, seeds, device=dev, dtype=dtype)[0])
    tiles_t = torch.stack(tiles, dim=0)
    rec = blend(tiles_t, tb, 64, 64)
    log.info("Blend result -> shape=%s dtype=%s stats(min=%.4f max=%.4f mean=%.4f)",
             tuple(rec.shape), rec.dtype, rec.min().item(), rec.max().item(), rec.mean().item())
    assert rec.shape == (1,1,64,64)

    # 4) Metrics API
    p = psnr(rec, rec)
    s = ssim(rec, rec)
    log.info("Metrics self-compare: psnr=%s ssim=%s", p, s)
    assert torch.isfinite(p).all() and torch.isfinite(s).all()
    assert (s >= 0.99).all(), "SSIM should be ~1 on identical images"
