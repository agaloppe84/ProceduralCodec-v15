# [STORE:OVERWRITE] — sérialisation binaire des records (v15)
from __future__ import annotations
import io, struct, zlib
from typing import List
from .records import TileRec

_LE = "<"  # little-endian

def _crc32(b: bytes) -> int:
    return zlib.crc32(b) & 0xFFFFFFFF

def pack_records_v15(recs: List[TileRec]) -> bytes:
    """Packe une liste de TileRec → bytes (v15)."""
    buf = io.BytesIO()
    buf.write(struct.pack(_LE + "I", len(recs)))  # record_count
    for r in recs:
        head = struct.pack(
            _LE + "I H H I H B I",
            r.tile_id, r.gen_id, r.qv_id, r.seed, r.rec_flags, r.payload_fmt, len(r.payload),
        )
        buf.write(head)
        buf.write(r.payload)
        crc = _crc32(head + r.payload)
        buf.write(struct.pack(_LE + "I", crc))  # rec_crc32
    return buf.getvalue()

def unpack_records_v15(b: bytes) -> List[TileRec]:
    """Parse bytes → liste de TileRec (v15) avec vérif CRC par record."""
    s = memoryview(b)
    if len(s) < 4:
        raise ValueError("records: too short")
    (count,) = struct.unpack_from(_LE + "I", s, 0)
    off = 4
    out: List[TileRec] = []
    for _ in range(count):
        need = 4+2+2+4+2+1+4  # header record fixe
        if off + need > len(s):
            raise ValueError("records: truncated header")
        tile_id, gen_id, qv_id, seed, rec_flags, payload_fmt, payload_len = struct.unpack_from(
            _LE + "I H H I H B I", s, off
        )
        off += need
        if off + payload_len + 4 > len(s):
            raise ValueError("records: truncated payload/crc")
        payload = bytes(s[off:off+payload_len])
        off += payload_len
        (crc_stored,) = struct.unpack_from(_LE + "I", s, off)
        off += 4
        head = struct.pack(_LE + "I H H I H B I", tile_id, gen_id, qv_id, seed, rec_flags, payload_fmt, payload_len)
        from_bytes_crc = _crc32(head + payload)
        if from_bytes_crc != crc_stored:
            raise ValueError("records: CRC mismatch")
        out.append(TileRec(tile_id, gen_id, qv_id, seed, rec_flags, payload_fmt, payload))
    if off != len(s):
        raise ValueError("records: trailing bytes")
    return out
