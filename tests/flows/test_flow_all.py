# tests/flows/test_flow_all.py
from __future__ import annotations
import importlib
import os
import torch

from pc15proc.register_all import register_all
from pc15proc.registry import get
from pc15proc.params import ParamCodec
from pc15codec.tiling import TileGridCfg, tile_image, blend
from pc15metrics import psnr, ssim
from pc15codec.bitstream import (
    pack_v15, unpack_v15,
    TileRec,
    pack_records_v15, unpack_records_v15,
    write_stream_v15, read_stream_v15,
)
from pc15codec.payload import RAW_FMT, encode_tile_payload_raw, decode_tile_payload_raw


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    assert os.getenv("PC15_ALLOW_CPU_TESTS", "0") == "1", "Set PC15_ALLOW_CPU_TESTS=1 for CPU runners"
    return torch.device("cpu")


def test_flow_core():
    """Flow 1 — Core: generators + tiling/blend + metrics (smoke)."""
    dev = _device()
    dtype = torch.float16 if dev.type == "cuda" else torch.float32

    register_all(verbose=False)
    g = get("STRIPES")
    pc = ParamCodec(g.info)
    P = pc.to_tensor({"freq": 32.25, "angle_deg": 90.0, "phase": 3.14159}, device=dev, dtype=dtype).unsqueeze(0)
    seeds = torch.tensor([0], device=dev, dtype=torch.int64)

    y = g.render((64, 64), P, seeds, device=dev, dtype=dtype)
    assert y.shape == (1, 1, 64, 64) and torch.isfinite(y).all()

    grid = TileGridCfg(size=32, overlap=8)
    spec = tile_image(y, grid)
    tiles = [g.render((grid.size, grid.size), P, seeds, device=dev, dtype=dtype)[0] for _ in range(spec.count)]
    tiles_t = torch.stack(tiles, dim=0)
    rec = blend(tiles_t, spec, 64, 64)
    assert rec.shape == y.shape and torch.isfinite(rec).all()

    ps, ss = psnr(y, y), ssim(y, y)
    assert torch.isfinite(ps).all() and torch.isfinite(ss).all()


def test_flow_bitstream_v15_raw():
    """Flow 2 — Bitstream v15: header + records RAW + stream (round-trips)."""
    hdr = {"width": 64, "height": 32, "tile": 16, "overlap": 8, "flags": 0, "meta": {"seed": 1}}
    b = pack_v15(hdr)
    h2 = unpack_v15(b)
    assert h2["width"] == 64 and h2["tile"] == 16

    fmt0, p0 = encode_tile_payload_raw(b"raw0")
    fmt1, p1 = encode_tile_payload_raw(b"raw1")
    assert fmt0 == fmt1 == RAW_FMT == 1

    recs = [
        TileRec(tile_id=0, gen_id=7, qv_id=3, seed=42, rec_flags=0, payload_fmt=fmt0, payload=p0),
        TileRec(tile_id=1, gen_id=7, qv_id=3, seed=43, rec_flags=1, payload_fmt=fmt1, payload=p1),
    ]
    blob = pack_records_v15(recs)
    r2 = unpack_records_v15(blob)
    assert len(r2) == 2 and r2[1].rec_flags == 1 and r2[0].payload == b"raw0"

    bs = write_stream_v15(hdr, recs)
    h3, r3 = read_stream_v15(bs)
    assert h3["width"] == 64 and len(r3) == 2
    assert decode_tile_payload_raw(r3[0].payload_fmt, r3[0].payload) == b"raw0"


def test_flow_public_api():
    """Flow 3 — Façade publique `pc15` (import / namespaces)."""
    pc = importlib.import_module("pc15")
    assert hasattr(pc, "__version__")
    for name in ("codec", "proc", "metrics", "data", "viz", "wf"):
        assert hasattr(pc, name)
