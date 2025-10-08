# packages/pc15codec/src/pc15codec/codec.py
from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, List, Tuple

import torch

from .config import CodecConfig
from .bitstream import (
    TileRec,
    read_stream_v15,
    write_stream_v15,
)
from .payload import (
    RAW_FMT,
    ANS0_FMT,
    encode_tile_payload_raw,
    decode_tile_payload_raw,
    encode_tile_payload,
    decode_tile_payload,
)
from .symbols import pack_symbols
from .rans import DEFAULT_TABLE_ID, load_table_by_id
from .tiling import TileGridCfg, tile_image, blend
from .search import SearchCfg, score_batch_bits
from .rd import estimate_bits_from_table

__all__ = ["CodecConfig", "encode_y", "decode_y"]


# ---------------------------------------------------------------------------
# ENV helpers
# ---------------------------------------------------------------------------

def _payload_mode_from_env() -> str:
    """
    Lit le mode payload (ENV `PC15_PAYLOAD_FMT`).

    - "RAW" → payloads bruts (debug / compat)
    - sinon → "ANS0" (défaut), i.e., rANS avec tables gelées référencées.
    """
    v = os.getenv("PC15_PAYLOAD_FMT", "ANS0").strip().upper()
    return "RAW" if v == "RAW" else "ANS0"


def _bits_mode_from_env() -> str:
    """
    Lit le mode d'estimation des bits (ENV `PC15_SCORE_BITS`).

    - "exact" → bits **exact** via encodage rANS (coûteux si beaucoup de candidats)
    - sinon   → "table" (défaut) : somme des -log2(p) à partir de la table gelée
    """
    v = os.getenv("PC15_SCORE_BITS", "table").strip().lower()
    return "exact" if v == "exact" else "table"


# ---------------------------------------------------------------------------
# Deterministic seed per tile (meilleure “dispersion” qu’un simple +tid)
# ---------------------------------------------------------------------------

def _tile_seed(global_seed: int, tile_id: int) -> int:
    """
    Mélange déterministe pour produire une seed par tuile dans [0..2^32-1].

    Formule: splitmix-like (petit hash 32-bit suffisant ici).
    """
    x = (int(global_seed) & 0xFFFFFFFF) ^ (0x9E3779B1 * int(tile_id))
    x ^= (x >> 16); x = (x * 0x85EBCA6B) & 0xFFFFFFFF
    x ^= (x >> 13); x = (x * 0xC2B2AE35) & 0xFFFFFFFF
    x ^= (x >> 16)
    return x & 0xFFFFFFFF


# ---------------------------------------------------------------------------
# Synthèse CPU (fallback) — utilisée AU DECODE (Step 4) et pour EVAL candidats
# ---------------------------------------------------------------------------

def _synth_stripes(H: int, W: int, *, freq: int, ang_deg: int, phase_deg: int,
                   device: str = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Tuile synthétique déterministe type “stripes” en [-1,1], shape [1,1,H,W].

    Paramètres discrets :
      - freq ∈ Z+ (densité de bandes)
      - ang_deg ∈ [0..179]
      - phase_deg ∈ [0..359] (ici on l’utilise en degrés, remappé en radians)
    """
    dev = torch.device(device)
    yy = torch.linspace(-1.0, 1.0, steps=H, device=dev, dtype=dtype)
    xx = torch.linspace(-1.0, 1.0, steps=W, device=dev, dtype=dtype)
    Y, X = torch.meshgrid(yy, xx, indexing="ij")

    ang = math.radians(int(ang_deg) % 180)
    phase = math.radians(int(phase_deg) % 360)
    u = torch.cos(2.0 * math.pi * int(freq) * (X * math.cos(ang) + Y * math.sin(ang)) + phase)
    return u.clamp(-1, 1).unsqueeze(0).unsqueeze(0)


def _synth_from_seed(H: int, W: int, *, seed: int,
                     device: str = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Variante sans paramètres explicites (utilisée pour la recon Step 4 “non nulle”).
    On déduit (freq,angle,phase) simples depuis la seed pour varier légèrement.
    """
    s = int(seed) & 0xFFFFFFFF
    freq = 3 + (s % 9)              # 3..11
    ang_deg = (s // 7) % 180
    phase_deg = (s >> 8) % 360
    return _synth_stripes(H, W, freq=freq, ang_deg=ang_deg, phase_deg=phase_deg,
                          device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# ENCODE — Step 5: “coarse RD search” (candidats synthétiques) + ANS0
# ---------------------------------------------------------------------------

def encode_y(img_y: torch.Tensor, cfg: CodecConfig) -> Dict[str, Any]:
    """
    Encode un plan Y en flux v15 avec **recherche coarse RD par tuile** + payload ANS0.

    Comportement
    ------------
    - Par tuile:
      1) on forme une petite **grille de candidats** (stripes : freq×angle×phase),
      2) on synthétise le batch [B,1,s,s] (CPU),
      3) on calcule la **distorsion D** (metric SearchCfg ; défaut "ssim"),
      4) on estime le **coût bits R** par candidat (ENV `PC15_SCORE_BITS` : "exact" ou "table"),
      5) on score `RD = D + λ·R` et on choisit le **meilleur**,
      6) on **quantize** les paramètres en offsets *discrets* (petits indices),
      7) on emballe en **ANS0** via `encode_tile_payload(...)`.
    - Le **framing v15** (header/records) est inchangé.
    - Le mode **RAW** reste disponible via `PC15_PAYLOAD_FMT=RAW` (debug).

    Retour
    ------
    dict: {"bitstream": bytes, "bpp": float, "stats": dict, "tile_map": []}
      - stats["tiles_info"] trace (tid, choix, score, bits, D)
    """
    assert img_y.ndim == 4 and img_y.shape[:2] == (1, 1), "expected [1,1,H,W]"
    H, W = int(img_y.shape[2]), int(img_y.shape[3])

    mode = _payload_mode_from_env()
    bits_mode = _bits_mode_from_env()
    use_raw = (mode == "RAW")

    hdr = {
        "width": W, "height": H,
        "tile": int(cfg.tile), "overlap": int(cfg.overlap),
        "flags": 0,
        "meta": {
            "encoder": "pc15codec@v15.0.0",
            "seed": int(cfg.seed),
            "ts": int(time.time()),
            "cfg": {
                "lambda": cfg.lambda_rd, "alpha": cfg.alpha_mix,
                "payload_mode": mode,
                "bits_mode": bits_mode,
                "table_id": cfg.rans_table_id or DEFAULT_TABLE_ID,
            },
        },
    }

    # Grille synthétique de candidats “coarse” (petite pour rester rapide CPU)
    # Offsets **discrets** pour rester dans l'intervalle 0..127 (safe pour pack_symbols)
    FREQS  = [4, 7, 10]                        # -> i_f ∈ {0,1,2}
    ANGLES = [0, 30, 60, 90, 120, 150]         # -> i_a ∈ {0..5}
    PHASES = [0, 90, 180, 270]                 # -> i_p ∈ {0..3}

    # Cache table rANS si besoin (mode "table")
    table = None
    if bits_mode == "table":
        table = load_table_by_id(cfg.rans_table_id or DEFAULT_TABLE_ID)

    # Config scoring
    scfg = SearchCfg(lambda_rd=float(cfg.lambda_rd), metric="ssim", alpha=float(cfg.alpha_mix))

    # Tiling
    rects: List[tuple[int, int, int, int]] = _grid_rects(H, W, cfg.tile, cfg.overlap)
    recs: List[TileRec] = []
    tiles_info: List[Dict[str, Any]] = []

    for tid, (y0, y1, x0, x1) in enumerate(rects):
        tile_seed = _tile_seed(cfg.seed, tid)

        if use_raw:
            # Mode debug/compat : payload RAW pass-through
            fmt, payload = encode_tile_payload_raw(b"raw")
            recs.append(TileRec(tile_id=tid, gen_id=0, qv_id=0, seed=tile_seed,
                                rec_flags=0, payload_fmt=fmt, payload=payload))
            tiles_info.append({"tid": tid, "mode": "RAW"})
            continue

        # 1) Vue tuile
        tile_y = img_y[..., y0:y1, x0:x1].to(device="cpu", dtype=torch.float32)

        # 2) Batch synth de tous les candidats
        synth_tiles = []
        cand_offsets = []   # triplet (i_f, i_a, i_p)
        for i_f, f in enumerate(FREQS):
            for i_a, a in enumerate(ANGLES):
                for i_p, p in enumerate(PHASES):
                    synth_tiles.append(_synth_stripes(tile_y.shape[-2], tile_y.shape[-1],
                                                      freq=f, ang_deg=a, phase_deg=p))
                    cand_offsets.append([i_f, i_a, i_p])
        synth = torch.cat(synth_tiles, dim=0)  # [B,1,s,s]
        B = synth.shape[0]

        # 3) Bits par candidat
        if bits_mode == "exact":
            # On encode symboliquement chaque candidat pour mesurer les bits
            bits_list: List[float] = []
            for offs in cand_offsets:
                _, blob = encode_tile_payload(
                    gen_id=0, qv_id=0, seed=tile_seed, flags=0,
                    offsets=offs, table_id=(cfg.rans_table_id or DEFAULT_TABLE_ID)
                )
                bits_list.append(float(len(blob) * 8))
            bits_tensor = torch.tensor(bits_list, dtype=synth.dtype)
        else:
            # "table": estimation via -log2(p) à partir de la table gelée
            bits_list: List[float] = []
            for offs in cand_offsets:
                syms = pack_symbols(0, 0, tile_seed, 0, offs)
                bits_list.append(estimate_bits_from_table(syms, table))  # somme -log2(p)
            bits_tensor = torch.tensor(bits_list, dtype=synth.dtype)

        # 4) D + RD
        out = score_batch_bits(tile_y, synth, scfg, bits=bits_tensor)
        RD = out.RD  # (B,)

        # 5) Choix meilleur candidat
        b_idx = int(torch.argmin(RD).item())
        best_offs = cand_offsets[b_idx]
        best_D = float(out.D[b_idx].item())
        best_bits = float(bits_tensor[b_idx].item())
        best_score = float(RD[b_idx].item())

        # 6) Pack & payload ANS0
        fmt, payload = encode_tile_payload(
            gen_id=0, qv_id=0, seed=tile_seed, flags=0,
            offsets=best_offs, table_id=(cfg.rans_table_id or DEFAULT_TABLE_ID)
        )

        recs.append(TileRec(tile_id=tid, gen_id=0, qv_id=0, seed=tile_seed,
                            rec_flags=0, payload_fmt=fmt, payload=payload))
        tiles_info.append({
            "tid": tid,
            "offs": best_offs, "D": best_D, "bits": best_bits, "RD": best_score,
        })

    # Écriture bitstream + stats
    blob = write_stream_v15(hdr, recs)
    bpp = (len(blob) * 8.0) / float(max(1, H * W))
    stats = {
        "tiles": len(recs),
        "payload_mode": mode,
        "bits_mode": bits_mode,
        "table_id": cfg.rans_table_id or DEFAULT_TABLE_ID,
        "lambda": cfg.lambda_rd, "alpha": cfg.alpha_mix,
        "bpp": bpp,
        "tiles_info": tiles_info,
    }
    return {"bitstream": blob, "bpp": bpp, "stats": stats, "tile_map": []}


# ---------------------------------------------------------------------------
# DECODE — Step 4 (inchangé) : recon non nulle via synthèse CPU + blend
# ---------------------------------------------------------------------------

def _grid_rects(H: int, W: int, tile: int, overlap: int) -> List[tuple[int, int, int, int]]:
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


def decode_y(bitstream: bytes, device: str = "cpu") -> torch.Tensor:
    """
    Décode un flux v15 et **reconstruit** une image Y non nulle (Step 4).

    - Valide ANS0/RAW par record.
    - Synthèse CPU-only par tuile (motif stripes basé sur la seed).
    - Assemblage via fenêtre Hann (`tiling.blend`).

    NB : on ignore encore le *contenu* symbolique pour la synthèse (les offsets
    choisis à l’encodage ne sont pas reconsommés ici — ce sera l’objet d’un step
    ultérieur quand le rendu “réel” sera branché).
    """
    header, records = read_stream_v15(bitstream)
    H, W = int(header["height"]), int(header["width"])
    size = int(header.get("tile", 256))
    overlap = int(header.get("overlap", 24))

    # Validation des payloads
    for r in records:
        if r.payload_fmt == ANS0_FMT:
            _ = decode_tile_payload(r.payload)
        elif r.payload_fmt == RAW_FMT:
            _ = decode_tile_payload_raw(r.payload_fmt, r.payload)

    # Grille/spec
    dummy = torch.zeros((1, 1, H, W), dtype=torch.float32, device="cpu")
    grid = TileGridCfg(size=size, overlap=overlap)
    spec = tile_image(dummy, grid)
    N = min(spec.count, len(records))

    tiles = []
    for i in range(N):
        seed = int(records[i].seed) & 0xFFFFFFFF
        t = _synth_from_seed(size, size, seed=seed, device="cpu", dtype=torch.float32)
        tiles.append(t[0])  # [1,1,s,s] -> [1,s,s]
    if not tiles:
        return torch.zeros((1, 1, H, W), dtype=torch.float32)

    tiles_tensor = torch.stack(tiles, dim=0)  # [N,1,s,s]
    recon = blend(tiles_tensor, spec, H, W, window="hann")
    return recon.to(dtype=torch.float32)
