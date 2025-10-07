# packages/pc15codec/src/pc15codec/payload.py
# -----------------------------------------------------------------------------
# PC15 payload helpers (Step 1 ready, Step 3 ANS0 formalisation)
# -----------------------------------------------------------------------------
# [ML/ENTROPY:WILL_STORE] - ce module manipule des données encodées (symb./tables)
# et fait partie du chemin de stockage entropique.
from __future__ import annotations
from typing import List, Tuple

# Convention bitstream v15 actuelle:
#  - 0 = rANS symbols (ANS0)
#  - 1 = RAW passthrough
# NE PAS inverser ces valeurs (cf. TileRec.payload_fmt).  # see bitstream/records spec
RAW_FMT: int = 1
ANS0_FMT: int = 0

# --- RAW passthrough (utile Step 1 / debug) ----------------------------------

def encode_tile_payload_raw(data: bytes) -> Tuple[int, bytes]:
    """Encode RAW -> (fmt, payload). Pas de compression.
    [STORE:OVERWRITE] (le caller écrit dans le bitstream)
    """
    return RAW_FMT, data

def decode_tile_payload_raw(fmt: int, payload: bytes) -> bytes:
    """Decode RAW <- (fmt, payload)."""
    if fmt != RAW_FMT:
        raise ValueError("decode_tile_payload_raw: wrong fmt (expected RAW_FMT=1)")
    return payload

# --- rANS (ANS0) chemin actuel (déjà présent dans ta base) -------------------
# Ces fonctions encodent/décodent les SYMBOLS (gen/qv/seed/flags/offsets) via rANS,
# et retournent/consomment un blob 'ANS0...' conforme à la spec interne.

from .symbols import pack_symbols, unpack_symbols            # [ML/ENTROPY:WILL_STORE]
from .rans import build_rans_tables, rans_encode, rans_decode # [ML/ENTROPY:WILL_STORE]

def encode_tile_payload(gen_id: int, qv_id: int, seed: int, flags: int,
                        offsets: List[int], precision: int = 12) -> bytes:
    """(gen_id,qv_id,seed,flags,offsets) -> symbols -> rANS -> payload bytes (b'ANS0'...)."""
    syms = pack_symbols(gen_id, qv_id, seed, flags, offsets)
    tables = build_rans_tables(syms, precision=precision)
    return rans_encode(syms, tables)

def decode_tile_payload(payload: bytes) -> Tuple[int, int, int, int, List[int]]:
    """payload bytes -> symbols -> (gen_id,qv_id,seed,flags,offsets)."""
    syms = rans_decode(payload, None)  # tables embarquées
    return unpack_symbols(syms)

__all__ = [
    "RAW_FMT", "ANS0_FMT",
    "encode_tile_payload_raw", "decode_tile_payload_raw",
    "encode_tile_payload", "decode_tile_payload",
]
