# packages/pc15codec/src/pc15codec/payload.py
# -----------------------------------------------------------------------------
# Step 2: RAW-only payload; rANS (ANS0) arrivera au Step 3.
# [ML/ENTROPY:WILL_STORE] - manipule des blobs encodés destinés au bitstream.
from __future__ import annotations
from typing import Tuple

# Formats de payload (convention PC15 v15)
ANS0_FMT = 0  # rANS avec tables embarquées (réservé Step 3)
RAW_FMT  = 1  # RAW passthrough (Step 2)

def encode_tile_payload_raw(data: bytes) -> Tuple[int, bytes]:
    """Encode RAW → (fmt, payload)."""
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("encode_tile_payload_raw: data must be bytes")
    return RAW_FMT, bytes(data)

def decode_tile_payload_raw(fmt: int, payload: bytes) -> bytes:
    """Decode (fmt,payload) quand fmt=RAW_FMT."""
    if fmt != RAW_FMT:
        raise ValueError("decode_tile_payload_raw: wrong fmt (expected RAW_FMT=1)")
    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError("decode_tile_payload_raw: payload must be bytes")
    return bytes(payload)

# Placeholders Step 3 (rANS)
def encode_tile_payload_ans0(data: bytes, table_id: str = "v15_default"):
    """Step 3 : rANS. Non disponible au Step 2."""
    raise NotImplementedError("ANS0 payload arrives in Step 3")

def decode_tile_payload_ans0(payload: bytes, table_id: str = "v15_default"):
    """Step 3 : rANS. Non disponible au Step 2."""
    raise NotImplementedError("ANS0 payload arrives in Step 3")

__all__ = [
    "RAW_FMT", "ANS0_FMT",
    "encode_tile_payload_raw", "decode_tile_payload_raw",
    "encode_tile_payload_ans0", "decode_tile_payload_ans0",
]
