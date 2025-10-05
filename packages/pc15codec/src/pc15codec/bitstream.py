from __future__ import annotations
import struct, zlib
from pc15core.errors import BitstreamError

_MAGIC = b"PC15"
_VERSION = 15

class BitstreamWriter:
    def __init__(self) -> None:
        self.chunks: list[bytes] = []
    def header(self, W: int, H: int, tile_size: int, overlap: int) -> None:
        head = struct.pack("<4sHIIHH", _MAGIC, _VERSION, W, H, tile_size, overlap)
        crc = zlib.crc32(head) & 0xFFFFFFFF
        self.chunks.append(head + struct.pack("<I", crc))
    def tile_record(self, tile_id: int, gen_id: int, qv_id: int, seed: int, flags: int, payload: bytes=b"") -> None:
        rec = struct.pack("<IHHQH", tile_id, gen_id, qv_id, seed & 0xFFFFFFFFFFFFFFFF, flags)
        rec += _varint(len(payload)) + payload
        self.chunks.append(rec)
    def finish(self) -> bytes:
        return b"".join(self.chunks)

class BitstreamReader:
    def __init__(self, data: bytes) -> None:
        self.data = memoryview(data)
        self.pos = 0
        self._read_header()
    def _read(self, n: int) -> bytes:
        if self.pos + n > len(self.data):
            raise BitstreamError("EOF")
        b = self.data[self.pos:self.pos+n]
        self.pos += n
        return bytes(b)
    def _read_header(self) -> None:
        magic, ver, W, H, ts, ov = struct.unpack("<4sHIIHH", self._read(4+2+4+4+2+2))
        if magic != _MAGIC or ver != _VERSION:
            raise BitstreamError("Header invalide")
        (crc,) = struct.unpack("<I", self._read(4))
        head = struct.pack("<4sHIIHH", magic, ver, W, H, ts, ov)
        if (zlib.crc32(head) & 0xFFFFFFFF) != crc:
            raise BitstreamError("CRC header invalide")
        self.W, self.H, self.tile_size, self.overlap = W, H, ts, ov

def _varint(n: int) -> bytes:
    out = bytearray()
    x = n
    while True:
        b = x & 0x7F
        x >>= 7
        if x:
            out.append(0x80 | b)
        else:
            out.append(b)
            break
    return bytes(out)
