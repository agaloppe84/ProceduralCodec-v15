from __future__ import annotations
from typing import List, Tuple

from .symbols import pack_symbols, unpack_symbols
from .rans import build_rans_tables, rans_encode, rans_decode

def encode_tile_payload(gen_id: int, qv_id: int, seed: int, flags: int, offsets: List[int], precision: int = 12) -> bytes:
    """(gen_id,qv_id,seed,flags,offsets) -> symbols -> rANS -> payload bytes (b'ANS1'...)."""
    syms = pack_symbols(gen_id, qv_id, seed, flags, offsets)
    tables = build_rans_tables(syms, precision=precision)
    return rans_encode(syms, tables)

def decode_tile_payload(payload: bytes) -> Tuple[int,int,int,int,List[int]]:
    """payload bytes -> symbols -> (gen_id,qv_id,seed,flags,offsets)."""
    syms = rans_decode(payload, None)  # tables embarquées
    return unpack_symbols(syms)
