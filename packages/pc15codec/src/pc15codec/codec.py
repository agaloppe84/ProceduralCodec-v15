# packages/pc15codec/src/pc15codec/codec.py
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List

import torch

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


__all__ = ["CodecConfig", "encode_y", "decode_y"]


@dataclass
class CodecConfig:
    """
    Configuration du codec Y-only v15 (Step 3).

    Attributs
    ---------
    tile : int
        Taille de tuile carrée (pixels).
    overlap : int
        Overlap (pixels) pour le blend inter-tuiles.
    colorspace : int
        0 = Y-only (Step 3), chroma viendra ensuite.
    flags : int
        Flags codec (réservé).
    payload_precision : int
        Précision des tables (bits) pour le rANS *si* génération à la volée.
        (Ici on utilise des tables gelées, champ conservé pour la méta.)
    lambda_rd : float
        Poids du coût bits dans le score RD (utilisé aux Steps suivants).
    alpha_mix : float
        Poids SSIM vs MSE dans la distortion (Steps suivants).
    seed : int
        Graine globale (déterminisme).
    """
    tile: int = 256
    overlap: int = 24
    colorspace: int = 0    # 0 = Y-only
    flags: int = 0
    payload_precision: int = 12
    lambda_rd: float = 0.015
    alpha_mix: float = 0.7
    seed: int = 1234


def _grid_rects(H: int, W: int, tile: int, overlap: int) -> List[tuple[int, int, int, int]]:
    """
    Génère les rectangles de tuiles avec overlap (ordre raster).

    Retourne une liste de tuples (y0, y1, x0, x1), *clampés* aux bords.
    """
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
    """
    Lit le mode payload depuis l’ENV `PC15_PAYLOAD_FMT`.
    - "RAW" : force payload RAW (debug / compat)
    - sinon : ANS0 (par défaut)
    """
    v = os.getenv("PC15_PAYLOAD_FMT", "ANS0").strip().upper()
    return "RAW" if v == "RAW" else "ANS0"


def encode_y(img_y: torch.Tensor, cfg: CodecConfig) -> Dict[str, Any]:
    """
    Encode un plan Y en flux v15 **avec payloads ANS0** (rANS) par défaut.

    Comportement Step 3
    -------------------
    - Framing v15 **inchangé** (header/records/stream).
    - Chaque tuile reçoit un `TileRec.payload_fmt`:
        * `ANS0_FMT` (0) si `PC15_PAYLOAD_FMT != "RAW"`
        * `RAW_FMT`  (1) si `PC15_PAYLOAD_FMT == "RAW"` (mode debug)
    - Le contenu ANS0 encode la **5-uplet symbolique** (gen_id, qv_id, seed, flags, offsets),
      ici encore “stub” (gen_id=0, qv_id=0, offsets=[]), l’objectif étant la **décodabilité déterministe**.
    - Aucune reconstruction n’est faite ici (Step 4 branchera la synthèse réelle).

    Retour
    ------
    dict avec :
      - "bitstream": bytes .pc15
      - "bpp": float (bits/pixel)
      - "stats": dict (infos utiles)
      - "tile_map": list (réservé)

    Exceptions
    ----------
    AssertionError si `img_y` n’est pas au format [1,1,H,W].
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
        # Seed dérivée par tuile pour garantir le déterminisme (mêmes cfg → mêmes bytes)
        tile_seed = (int(cfg.seed) + tid) & 0xFFFFFFFF

        if use_raw:
            # Mode debug/compat : payload RAW pass-through
            fmt, payload = encode_tile_payload_raw(b"raw")
            gen_id = 0
            qv_id = 0
        else:
            # Step 3 : encodage ANS0 symbolique “minimal viable”
            # offsets=[] tant que la recherche RD n’est pas appliquée (Step 4+)
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


def decode_y(bitstream: bytes, device: str = "cuda") -> torch.Tensor:
    """
    Décode un flux v15, **valide** les payloads ANS0/RAW et renvoie une image Y noire.

    Comportement Step 3
    -------------------
    - On lit header + records via `read_stream_v15`.
    - Pour chaque record :
        * si `payload_fmt == ANS0_FMT` → `decode_tile_payload(...)`
          (valide l’entête ANS0, charge la table et reconstitue les symboles).
        * si `payload_fmt == RAW_FMT`  → `decode_tile_payload_raw(...)`.
    - La véritable reconstruction par tuiles sera branchée au Step 4 ; ici on
      renvoie un tenseur noir de la bonne dimension pour valider le trajet I/O.

    Paramètres
    ----------
    bitstream : bytes
        Flux produit par `encode_y` (Step 3).
    device : str
        Réservé pour Step 4+ (CUDA vs CPU). Non utilisé ici.

    Retour
    ------
    torch.Tensor
        Tenseur [1,1,H,W] (float32), rempli de zéros.
    """
    header, records = read_stream_v15(bitstream)
    H, W = int(header["height"]), int(header["width"])

    for r in records:
        if r.payload_fmt == ANS0_FMT:
            _ = decode_tile_payload(r.payload)  # round-trip symbolique OK
        elif r.payload_fmt == RAW_FMT:
            _ = decode_tile_payload_raw(r.payload_fmt, r.payload)
        else:
            # Format inconnu : on ignore mais on pourrait lever si on veut strict
            pass

    # Step 3: pas encore de reconstruction → image noire (bonne forme)
    return torch.zeros((1, 1, H, W), dtype=torch.float32)
