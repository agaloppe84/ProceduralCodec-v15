from __future__ import annotations
from pathlib import Path

# Header helpers are v15-only at this stage; full stream will arrive later.
from .header import pack_v15, unpack_v15  # re-exported by package

def read_bitstream(path: str | Path) -> bytes:
    """Read a bitstream from disk (raw bytes)."""
    return Path(path).read_bytes()

def write_bitstream(payload: bytes, path: str | Path) -> None:
    """Atomic write to target path.  # [STORE:OVERWRITE]"""
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(p)
