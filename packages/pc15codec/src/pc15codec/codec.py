# packages/pc15codec/src/pc15codec/codec.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import math
import time

import torch

from .bitstream import Header, TileRec, write_bitstream, read_bitstream
from .payload import encode_tile_payload, decode_tile_payload

__all__ = ["CodecConfig", "encode_y", "decode_y"]

@dataclass
class CodecConfig:
    tile: int = 256
    overlap: int = 24
    payload_precision: int = 12
    colorspace: int = 0   # 0 = Y (luma-only, S1)
    flags: int = 0        # reserved
    # recherche / RD (placeholders pour S2+)
    lambda_rd: float = 0.015
    alpha_mix: float = 0.7
    # seeds / déterminisme
    seed: int = 1234

def _tile_grid(H: int, W: int, tile: int, overlap: int) -> List[Tuple[int,int,int,int]]:
    assert tile > 0 and overlap >= 0 and overlap < tile
    stride = tile - overlap
    ys = list(range(0, max(1, H - overlap), stride))
    xs = list(range(0, max(1, W - overlap), stride))
    rects: List[Tuple[int,int,int,int]] = []
    for y in ys:
        for x in xs:
            y1 = min(y + tile, H)
            x1 = min(x + tile, W)
            rects.append((y, y1, x, x1))
    return rects

def _blend_paste(dst: torch.Tensor, src: torch.Tensor, y0: int, y1: int, x0: int, x1: int, overlap: int) -> None:
    """S1: blend rectangle src (1,1,h,w) into dst (1,1,H,W) with simple cosine on edges."""
    h, w = y1 - y0, x1 - x0
    # poids 1 partout (S1 simplifié); on branchera une Hann 2D plus tard si besoin
    dst[..., y0:y1, x0:x1] = src

def encode_y(img_y: torch.Tensor, cfg: CodecConfig) -> Dict[str, Any]:
    """
    Encode Y en bitstream .pc15 (S1: symbole rANS par tuile, synthèse à venir).
    Retourne: {bitstream: bytes, bpp: float, stats: dict, tile_map: list}
    """
    assert img_y.ndim == 4 and img_y.shape[0] == 1 and img_y.shape[1] == 1, "expected [1,1,H,W]"
    H = int(img_y.shape[2]); W = int(img_y.shape[3])

    # Header (meta minimale — stable & déterministe)
    meta = {
        "encoder": "pc15codec@v15.0.0",
        "seed": int(cfg.seed),
        "ts": int(time.time()),
        "cfg": {
            "tile": cfg.tile, "overlap": cfg.overlap,
            "payload_precision": cfg.payload_precision,
            "lambda": cfg.lambda_rd, "alpha": cfg.alpha_mix,
        }
    }
    header = Header(width=W, height=H, tile=cfg.tile, overlap=cfg.overlap,
                    colorspace=cfg.colorspace, flags=cfg.flags, meta=meta)

    # Boucle tuiles — S1: on encode un payload de décisions symboliques "stub"
    rects = _tile_grid(H, W, cfg.tile, cfg.overlap)
    recs: List[TileRec] = []
    tile_map: List[Dict[str, int]] = []
    gen_id_stub = 0   # S2: remplacé par la recherche (id générateur)
    qv_id_stub  = 0   # S2: remplacé par l'index de codebook
    flags_stub  = 0   # S2: bit flags (résidu, etc.)
    for tid, (y0, y1, x0, x1) in enumerate(rects):
        seed_i = (cfg.seed + tid) & 0xFFFFFFFF
        offsets: List[int] = []  # S2: offsets QV bornés [-128,127]
        payload = encode_tile_payload(gen_id_stub, qv_id_stub, seed_i, flags_stub, offsets, precision=cfg.payload_precision)
        recs.append(TileRec(tile_id=tid, gen_id=gen_id_stub, qv_id=qv_id_stub,
                            seed=seed_i, rec_flags=flags_stub, payload_fmt=0, payload=payload))
        tile_map.append({"tile_id": tid, "y0": y0, "y1": y1, "x0": x0, "x1": x1})

    bitstream = write_bitstream(header, recs)
    bpp = (len(bitstream) * 8.0) / float(H * W) if (H > 0 and W > 0) else 0.0
    stats = {
        "tiles": len(recs),
        "payload_precision": cfg.payload_precision,
        "bpp": bpp,
    }
    return {"bitstream": bitstream, "bpp": bpp, "stats": stats, "tile_map": tile_map}

def decode_y(bitstream: bytes, device: str = "cuda") -> torch.Tensor:
    """
    Décode un bitstream .pc15 vers un tenseur [1,1,H,W] (S1 squelette).
    TODO S2+: utiliser (gen_id,qv_id,seed,offsets) pour synthétiser la tuile via pc15proc.render,
             puis blend Hann sur l'overlap. Pour l’instant, renvoie un canvas noir (fondations).
    """
    header, recs = read_bitstream(bitstream)
    H, W = int(header.height), int(header.width)
    tile, overlap = int(header.tile), int(header.overlap)

    # Canvas de sortie (noir pour S1; on branchera la synthèse dans S2)
    y = torch.zeros((1, 1, H, W), dtype=torch.float32)

    # Exemple: parcourir les tuiles et lire les payloads (démo parse)
    rects = _tile_grid(H, W, tile, overlap)
    for rec, (y0, y1, x0, x1) in zip(recs, rects):
        gen_id, qv_id, seed_i, flags, offsets = decode_tile_payload(rec.payload)
        # S2: synthèse via pc15proc + QV → tile_y
        tile_y = torch.zeros((1, 1, y1 - y0, x1 - x0), dtype=torch.float32)
        _blend_paste(y, tile_y, y0, y1, x0, x1, overlap)

    return y
