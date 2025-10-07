# [STORE:OVERWRITE] — framing complet v15: header (Step 0) + records (Step 2)
from __future__ import annotations
import struct
from typing import Dict, List, Tuple
from .header import pack_v15, unpack_v15
from .records import TileRec
from .records_io import pack_records_v15, unpack_records_v15

_LE = "<"

def _header_size_from_prefix(buf: bytes) -> int:
    """Calcule la taille totale du header v15 (magic+version+len+json+crc) sans l'interpréter.
    Format attendu (LE): 'PC15'(4) | version(1) | header_len(u32)(4) | header_json(header_len) | header_crc32(u32)(4)
    """
    if len(buf) < 9:
        raise ValueError("buffer too short for header prefix")
    if buf[:4] != b"PC15":
        raise ValueError("bad magic")
    # version = buf[4] (on ne vérifie pas ici, 'unpack_v15' le fera si besoin)
    (header_len,) = struct.unpack_from(_LE + "I", buf, 5)
    total = 4 + 1 + 4 + header_len + 4
    if len(buf) < total:
        raise ValueError("buffer too short for full header")
    return total

def write_stream_v15(header: Dict, records: List[TileRec]) -> bytes:
    """Concatène header v15 + records v15 → bytes."""
    h = pack_v15(header)             # Step 0
    r = pack_records_v15(records)    # Step 2
    return h + r

def read_stream_v15(buf: bytes) -> Tuple[Dict, List[TileRec]]:
    """Découpe buf en (header, records) et parse chaque partie."""
    hdr_size = _header_size_from_prefix(buf)
    header = unpack_v15(buf[:hdr_size])   # Step 0
    recs   = unpack_records_v15(buf[hdr_size:])  # Step 2
    return header, recs
