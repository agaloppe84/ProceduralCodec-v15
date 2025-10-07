# packages/pc15codec/src/pc15codec/codec.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List
import time
import torch

# v15 bitstream (package)
from .bitstream import (
    pack_v15, unpack_v15,
    TileRec, write_stream_v15, read_stream_v15,
)

# Step 2: payload RAW-only
from .payload import RAW_FMT, encode_tile_payload_raw, decode_tile_payload_raw

__all__ = ["CodecConfig", "encode_y", "decode_y"]

@dataclass
class CodecConfig:
    tile: int = 256
    overlap: int = 24
    colorspace: int = 0    # 0 = Y-only
    flags: int = 0
    payload_precision: int = 12
    lambda_rd: float = 0.015
    alpha_mix: float = 0.7
    seed: int = 1234

def _grid_rects(H: int, W: int, tile: int, overlap: int) -> List[tuple[int,int,int,int]]:
    assert tile > 0 and 0 <= overlap < tile
    stride = tile - overlap
    ys = list(range(0, max(1, H - overlap), stride))
    xs = list(range(0, max(1, W - overlap), stride))
    rects: List[tuple[int,int,int,int]] = []
    for y0 in ys:
        for x0 in xs:
            y1 = min(y0 + tile, H)
            x1 = min(x0 + tile, W)
            rects.append((y0, y1, x0, x1))
    return rects

def encode_y(img_y: torch.Tensor, cfg: CodecConfig) -> Dict[str, Any]:
    """
    Step 2 skeleton:
     - header v15 (dict + pack_v15)
     - records v15 (RAW payload pass-through)
     - aucun rANS ici (ANS0 arrive au Step 3)
    """
    assert img_y.ndim == 4 and img_y.shape[:2] == (1, 1), "expected [1,1,H,W]"
    H, W = int(img_y.shape[2]), int(img_y.shape[3])

    hdr = {
        "width": W, "height": H,
        "tile": int(cfg.tile), "overlap": int(cfg.overlap),
        "flags": int(cfg.flags),
        "meta": {
            "encoder": "pc15codec@v15.0.0",
            "seed": int(cfg.seed),
            "ts": int(time.time()),
            "cfg": {
                "payload_precision": cfg.payload_precision,
                "lambda": cfg.lambda_rd, "alpha": cfg.alpha_mix,
            },
        },
    }

    rects = _grid_rects(H, W, cfg.tile, cfg.overlap)
    recs: List[TileRec] = []
    for tid, _ in enumerate(rects):
        # Step 2: payload “stub” en RAW (exemple)
        fmt, payload = encode_tile_payload_raw(b"raw")
        recs.append(TileRec(tile_id=tid, gen_id=0, qv_id=0, seed=(cfg.seed + tid) & 0xFFFFFFFF,
                            rec_flags=0, payload_fmt=fmt, payload=payload))

    blob = write_stream_v15(hdr, recs)
    bpp = (len(blob) * 8.0) / float(max(1, H * W))
    stats = {"tiles": len(recs), "payload_precision": cfg.payload_precision, "bpp": bpp}
    return {"bitstream": blob, "bpp": bpp, "stats": stats, "tile_map": []}

def decode_y(bitstream: bytes, device: str = "cuda") -> torch.Tensor:
    """
    Step 2 skeleton decode: lit le flux v15 (header+records), recon noir.
    """
    header, records = read_stream_v15(bitstream)
    H, W = int(header["height"]), int(header["width"])
    # Parcours d'exemple des payloads RAW
    for r in records:
        if r.payload_fmt == RAW_FMT:
            _ = decode_tile_payload_raw(r.payload_fmt, r.payload)
    return torch.zeros((1, 1, H, W), dtype=torch.float32)
