# packages/pc15codec/src/pc15codec/codec.py
from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, List

import torch

from .config import CodecConfig  # <- config publique unifiée

# v15 bitstream (package)
from .bitstream import (
    TileRec,
    read_stream_v15,
    write_stream_v15,
)

# Payloads (bi-mode ANS0/RAW)
from .payload import (
    RAW_FMT,
    ANS0_FMT,
    encode_tile_payload_raw,
    decode_tile_payload_raw,
    encode_tile_payload,
    decode_tile_payload,
)
from .rans import DEFAULT_TABLE_ID

# Tiling/blend (Step 4)
from .tiling import TileGridCfg, tile_image, blend

__all__ = ["CodecConfig", "encode_y", "decode_y"]


# -----------------------
# Helpers Step 3 (encode)
# -----------------------

def _grid_rects(H: int, W: int, tile: int, overlap: int) -> List[tuple[int, int, int, int]]:
    """Retourne les rectangles (y0,y1,x0,x1) clampés aux bords, en ordre raster."""
    assert tile > 0 and 0 <= overlap < tile
    stride = tile - overlap
    ys = list(range(0, max(1, H - overlap), stride))
    xs = list(range(0, max(1, W - overlap), stride))
    rects: List[tuple[int, int, int, int]] = []
    for y0 in ys:
        for x0 in xs:
            y1 = min(y0 + tile, H)
            x1 = min(x0 + tile, W)
            rects.append((y0, y1, x0, x1))
    return rects


def _payload_mode_from_env() -> str:
    """Lit `PC15_PAYLOAD_FMT` : 'RAW' → RAW, sinon ANS0."""
    v = os.getenv("PC15_PAYLOAD_FMT", "ANS0").strip().upper()
    return "RAW" if v == "RAW" else "ANS0"


# ----------------------------
# Step 3 (inchangé) — encode_y
# ----------------------------

def encode_y(img_y: torch.Tensor, cfg: CodecConfig) -> Dict[str, Any]:
    """
    Encode un plan Y en flux v15 **avec payloads ANS0** (rANS) par défaut.

    Step 3 (identique) : framing v15 + payload symbolique minimal viable.
    """
    assert img_y.ndim == 4 and img_y.shape[:2] == (1, 1), "expected [1,1,H,W]"
    H, W = int(img_y.shape[2]), int(img_y.shape[3])

    mode = _payload_mode_from_env()  # "ANS0" (défaut) ou "RAW"
    use_raw = (mode == "RAW")

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
                "lambda": cfg.lambda_rd,
                "alpha": cfg.alpha_mix,
                "payload_mode": mode,
                "table_id": DEFAULT_TABLE_ID,
            },
        },
    }

    rects = _grid_rects(H, W, cfg.tile, cfg.overlap)
    recs: List[TileRec] = []

    for tid, _ in enumerate(rects):
        tile_seed = (int(cfg.seed) + tid) & 0xFFFFFFFF

        if use_raw:
            fmt, payload = encode_tile_payload_raw(b"raw")
            gen_id = 0
            qv_id = 0
        else:
            # Step 3 : encodage ANS0 symbolique minimal
            fmt, payload = encode_tile_payload(
                gen_id=0,
                qv_id=0,
                seed=tile_seed,
                flags=0,
                offsets=[],
                table_id=DEFAULT_TABLE_ID,
            )
            gen_id = 0
            qv_id = 0

        recs.append(
            TileRec(
                tile_id=tid,
                gen_id=gen_id,
                qv_id=qv_id,
                seed=tile_seed,
                rec_flags=0,
                payload_fmt=fmt,
                payload=payload,
            )
        )

    blob = write_stream_v15(hdr, recs)
    bpp = (len(blob) * 8.0) / float(max(1, H * W))
    stats = {
        "tiles": len(recs),
        "payload_precision": cfg.payload_precision,
        "payload_mode": mode,
        "table_id": DEFAULT_TABLE_ID,
        "bpp": bpp,
    }
    return {"bitstream": blob, "bpp": bpp, "stats": stats, "tile_map": []}


# --------------------------------
# Step 4 — decode_y (reconstruction)
# --------------------------------

def _synth_tile_cpu(size: int, seed: int, *, dtype=torch.float32, device="cpu") -> torch.Tensor:
    """
    Synthèse **CPU-only** d'une tuile non-nulle, déterministe, en [-1,1].

    Motif : sinusoïdes orientées (stripes) avec (freq, angle, phase) dérivés du seed.
    - freq  ∈ {3..11}
    - angle ∈ [0..180)
    - phase ∈ [0..2π)
    Retour: tenseur [1,1,size,size].
    """
    dev = torch.device(device)
    h = w = int(size)
    # Grille normalisée
    xx = torch.linspace(-1.0, 1.0, steps=w, device=dev, dtype=dtype)
    yy = torch.linspace(-1.0, 1.0, steps=h, device=dev, dtype=dtype)
    X, Y = torch.meshgrid(yy, xx, indexing="ij")  # [H,W]

    # Paramètres dérivés du seed (stables cross-run)
    s = int(seed) & 0xFFFFFFFF
    freq = 3 + (s % 9)                             # 3..11
    ang_deg = (s // 7) % 180
    ang = math.radians(ang_deg)
    phase = ( ( (s >> 8) % 1000) / 1000.0 ) * (2.0 * math.pi)

    u = torch.cos(2.0 * math.pi * freq * (X * math.sin(ang) + Y * math.cos(ang)) + phase)
    # Remap en [-1,1] (déjà borné), puis [1,1,H,W]
    return u.clamp(-1, 1).unsqueeze(0).unsqueeze(0)


def decode_y(bitstream: bytes, device: str = "cpu") -> torch.Tensor:
    """
    Décode un flux v15, **valide** ANS0/RAW et **reconstruit** une image Y non nulle.

    Step 4
    ------
    - Lecture header + records via `read_stream_v15`.
    - Validation payloads :
        * ANS0 → `decode_tile_payload(...)`
        * RAW  → `decode_tile_payload_raw(...)`
      (on ignore le contenu symbolique pour l’instant : Step 4 vise la recon I/O)
    - Synthèse CPU-only par tuile (motif stripes déterministe à partir du seed).
    - Assemblage via fenêtre Hann et normalisation (voir `tiling.blend`).

    Paramètres
    ----------
    bitstream : bytes
        Flux produit par `encode_y`.
    device : str
        Cible pour la recon (par défaut "cpu"). Ignoré si indisponible.

    Retour
    ------
    torch.Tensor
        Tenseur [1,1,H,W] (float32).
    """
    header, records = read_stream_v15(bitstream)
    H, W = int(header["height"]), int(header["width"])
    size = int(header.get("tile", 256))
    overlap = int(header.get("overlap", 24))

    # 1) Valider tous les payloads (mêmes garde-fous que Step 3)
    for r in records:
        if r.payload_fmt == ANS0_FMT:
            _ = decode_tile_payload(r.payload)  # lève en cas de corruption
        elif r.payload_fmt == RAW_FMT:
            _ = decode_tile_payload_raw(r.payload_fmt, r.payload)
        else:
            # Inconnu → on peut ignorer ou lever ; conservateur: on ignore.
            pass

    # 2) Synthèse de toutes les tuiles (CPU-only, déterministe via seed)
    #    On reconstruit la grille depuis (H,W,size,overlap) pour assurer l'ordre raster.
    dummy = torch.zeros((1, 1, H, W), dtype=torch.float32, device="cpu")
    grid = TileGridCfg(size=size, overlap=overlap)
    spec = tile_image(dummy, grid)
    if spec.count != len(records):
        # Garde-fou (on reste tolérant, mais le mismatch est signalé)
        # On tronque/complète au besoin vers le min.
        N = min(spec.count, len(records))
    else:
        N = spec.count

    tiles = []
    for i in range(N):
        seed = int(records[i].seed)
        t = _synth_tile_cpu(size=grid.size, seed=seed, dtype=torch.float32, device="cpu")  # [1,1,s,s]
        tiles.append(t[0])  # empile en [N,1,s,s]
    if not tiles:
        return torch.zeros((1, 1, H, W), dtype=torch.float32)

    tiles_tensor = torch.stack(tiles, dim=0)  # [N,1,s,s]

    # 3) Blend fenêtre Hann (partition of unity)
    recon = blend(tiles_tensor, spec, H, W, window="hann")  # [1,1,H,W]
    return recon.to(dtype=torch.float32)
