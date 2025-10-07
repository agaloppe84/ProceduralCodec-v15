# packages/pc15codec/src/pc15codec/bitstream/__init__.py
from __future__ import annotations

# I/O bruts du bitstream (Step 0)
from .io import read_bitstream, write_bitstream          # [STORE:OVERWRITE]

# Header v15 (Step 0)
from .header import pack_v15, unpack_v15

# Records (Step 2)
from .records import TileRec
from .records_io import pack_records_v15, unpack_records_v15

# Framing complet header+records (Step 2)
from .stream import write_stream_v15, read_stream_v15

__all__ = [
    "read_bitstream", "write_bitstream",
    "pack_v15", "unpack_v15",
    "TileRec",
    "pack_records_v15", "unpack_records_v15",
    "write_stream_v15", "read_stream_v15",
]
