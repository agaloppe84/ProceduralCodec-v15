# packages/pc15codec/src/pc15codec/bitstream/stream.py
from __future__ import annotations
from typing import List, Tuple
import struct

from .records import TileRec
from .records_io import pack_records_v15, unpack_records_v15
from .header import MAGIC, VERSION, pack_v15, unpack_v15


def _peek_header_size(buf: bytes) -> int:
    """
    Retourne la taille totale du header v15 présent au début de buf, sans le décoder :
      total = 10 (magic+ver+len) + n (payload JSON) + 4 (CRC32)
    Lève ValueError si préfixe invalide ou buffer trop court.
    """
    if len(buf) < 10:
        raise ValueError("read_stream_v15: truncated buffer (need >= 10 bytes for header prefix)")
    magic = buf[:4]
    if magic != MAGIC:
        raise ValueError("read_stream_v15: bad magic")
    ver = struct.unpack(">H", buf[4:6])[0]
    if ver != VERSION:
        raise ValueError("read_stream_v15: bad version")
    n = struct.unpack(">I", buf[6:10])[0]
    total = 10 + int(n) + 4
    if len(buf) < total:
        raise ValueError("read_stream_v15: truncated header payload/CRC")
    return total


def write_stream_v15(header_dict: dict, records: List[TileRec]) -> bytes:
    """
    Concatène header v15 (pack_v15) + bloc records v15 (pack_records_v15).
    """
    h = pack_v15(header_dict)          # bytes
    r = pack_records_v15(records)      # bytes
    return h + r


def read_stream_v15(buf: bytes) -> Tuple[dict, List[TileRec]]:
    """
    Sépare et décode header v15 + bloc records v15 depuis un flux binaire unique.
    """
    hdr_size = _peek_header_size(buf)      # calcule 10+n+4 de manière sûre
    header = unpack_v15(buf[:hdr_size])    # decode header complet
    recs = unpack_records_v15(buf[hdr_size:])  # le reste = bloc records
    return header, recs
