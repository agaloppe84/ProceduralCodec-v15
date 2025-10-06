from __future__ import annotations
import json, struct, zlib

MAGIC = b"PC15"
VERSION = 15

# Header schema (v15) — JSON payload inside a framed binary with CRC32.
# Stable and versionable; payload keys are validated by caller.

def pack_v15(h: dict) -> bytes:
    """Pack header dict into bytes with MAGIC|VER|LEN|JSON|CRC.
    Returns bytes suitable for writing to disk.  # [STORE:OVERWRITE]
    """
    body = json.dumps(h, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    buf = MAGIC + struct.pack(">H", VERSION) + struct.pack(">I", len(body)) + body
    crc = zlib.crc32(buf) & 0xFFFFFFFF
    return buf + struct.pack(">I", crc)

def unpack_v15(b: bytes) -> dict:
    magic = b[:4]
    if magic != MAGIC: raise ValueError("Bad magic")
    ver = struct.unpack(">H", b[4:6])[0]
    if ver != VERSION: raise ValueError("Bad version")
    n = struct.unpack(">I", b[6:10])[0]
    payload = b[10:10+n]
    crc_stored = struct.unpack(">I", b[10+n:10+n+4])[0]
    if (zlib.crc32(b[:10+n]) & 0xFFFFFFFF) != crc_stored:
        raise ValueError("Header CRC mismatch")
    return json.loads(payload)
