
"""
PC15 — Procedural Codec v15 — Bitstream
=======================================

This module defines the **v15 bitstream** layout used by the PC15 procedural codec.
It is **decode-oriented** (only stores what is needed to reconstruct pixels) and
designed for **determinism, idempotence, and forward-compatibility**.

-------------------------------------------------------------------------------
# FULL PIPELINE SPEC (Context for the bitstream)
-------------------------------------------------------------------------------
The codec compresses an image **procedurally** by tiles, choosing for each tile
a procedural **generator** and quantized **parameters** (by **QV**), then entropy-
coding the symbolic choices with **rANS**. An optional sparse **residual** may be
added for out-of-class tiles. Reconstruction synthesizes tiles on GPU and blends
over the overlapping borders (Hann/cosine window).

Global high-level flow (encode):
  1) Tiling (size=T, overlap=O). Process luminance Y first, then 4:2:0 chroma.
  2) For each tile:
     2.1) Generate B candidates (generator + params + seed), batch on GPU.
     2.2) Evaluate RD:  D = α·(1-SSIM) + (1-α)·MSE_norm + seam_penalty
                        R ≈ bits of symbols (models for rANS + codebook indices).
          Early-exit if D < τ; optional beam K=2–4.
     2.3) Quantize params with QV: get codebook index + fine offsets.
     2.4) Emit symbols: (gen_id, qv_index, offset bins, flags).
     2.5) (Optional, capped) sparse residual if “off-class” tile.
  3) Entropy-code symbols with rANS (global or per-class tables).
  4) Write the bitstream (header → tile records → footer checksum).
  5) Name the file deterministically (dataset/img/codec/v15/q/bpp/seed).

Global high-level flow (decode):
  1) Parse header, validate version=15 and CRCs.
  2) For each tile record:
     2.1) Decode symbols with rANS → (gen_id, qv_index, offsets, flags).
     2.2) Dequantize params via codebook (QV).
     2.3) Synthesize tile procedurally (seed ensures determinism).
     2.4) If residual_present, add sparse residual.
  3) Reconstruct the full image, blend overlaps with Hann window.
  4) If chroma present (4:2:0), upsample and combine with Y.

-------------------------------------------------------------------------------
# BITSTREAM v15 (Binary)
-------------------------------------------------------------------------------

Little-endian. All fields aligned to 4 bytes unless otherwise stated.

HEADER (variable length, includes CRC32 at the end)
--------------------------------------------------
struct HeaderFixed {
    char magic[4]    = "PC15";         // 0x50 0x43 0x31 0x35
    uint16 version   = 15;             // 0x0F 0x00
    uint16 hdr_len;                    // bytes, includes CRC field
    uint32 width;                      // image width in pixels
    uint32 height;                     // image height in pixels
    uint16 tile;                       // tile size (e.g., 256)
    uint16 overlap;                    // overlap size (e.g., 24)
    uint8  colorspace;                 // 0 = Y-only, 1 = YCbCr 4:2:0
    uint8  flags;                      // bit0: residual_enabled; bit1: has_meta_json
    uint16 reserved;                   // set to 0
    uint32 meta_json_len;              // 0 if no meta
    // uint8  meta_json[meta_json_len]; // UTF-8 JSON (optional)
    // padding to 4-byte boundary
    // uint32 header_crc32;             // CRC32 of the header bytes excluding this field
};

- `meta_json` (optional) contains non-essential encode-time info, e.g.:
    {
      "encoder": "pc15codec@v15.0.0",
      "git": "abc123",
      "seed": 1234,
      "qv_models": [{"id":"qv_v15_luma","sha256":"..."}, ...],
      "rans_tables": [{"id":"rans_v15_global","sha256":"..."}]
    }

TILE RECORDS (concatenated)
---------------------------
struct TileRecord {
    uint32 tile_id;                    // linear tile index (scanline order)
    uint16 gen_id;                     // generator ID (from catalog)
    uint16 qv_id;                      // codebook ID (index in codebook table)
    uint32 seed;                       // RNG seed for this tile
    uint16 rec_flags;                  // bit0: residual_present
    uint16 payload_fmt;                // 0: rANS+symbols; 1: raw; (reserve range)
    uint32 payload_len;                // bytes
    // uint8 payload[payload_len];      // rANS blob or raw
    // padding to 4-byte boundary
};

- `payload` normally stores rANS-compressed symbols. For S1 skeleton this can
  be a **passthrough** (e.g., b"ANS0" + raw symbols).

FOOTER (fixed size, 12 bytes)
-----------------------------
struct Footer {
    uint32 tiles_count;
    uint32 reserved;       // 0
    uint32 global_crc32;   // CRC32 of all bytes BEFORE this field (header+records)
};

VALIDATION
----------
- Header: check magic, version, CRC32.
- Footer: recompute `global_crc32` and compare.
- Robustness: refuse to parse if any bound is exceeded or if sizes mismatch.

EXTENSION POINTS
----------------
- Add new `payload_fmt` values for future symbol/residual encodings.
- Add fields to `meta_json` without breaking decoding.
- When chroma is present (colorspace=1), append chroma tile records after Y.

-------------------------------------------------------------------------------
Implementation below
-------------------------------------------------------------------------------
"""
from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

MAGIC = b"PC15"
VERSION = 15
FOOTER_SIZE = 12  # tiles_count (u32), reserved (u32), global_crc32 (u32)

FIXED_FMT  = "<4sHHIIHHBBHI"
FIXED_SIZE = struct.calcsize(FIXED_FMT)  # = 28 bytes

TILE_HEAD_FMT  = "<IHHIHHI"
TILE_HEAD_SIZE = struct.calcsize(TILE_HEAD_FMT)  # = 20 bytes

# ----------------------------- helpers ------------------------------------

def _crc32(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF

def _pad4(n: int) -> int:
    return (4 - (n % 4)) % 4

# ----------------------------- data classes -------------------------------

@dataclass
class Header:
    width: int
    height: int
    tile: int
    overlap: int
    colorspace: int = 0  # 0=Y-only, 1=YCbCr 4:2:0
    flags: int = 0       # bit0: residual_enabled, bit1: has_meta_json
    meta: Dict[str, Any] | None = None

@dataclass
class TileRec:
    tile_id: int
    gen_id: int
    qv_id: int
    seed: int
    rec_flags: int = 0     # bit0: residual_present
    payload_fmt: int = 0   # 0=rANS symbols, 1=raw
    payload: bytes = b""

# ----------------------------- pack/unpack header -------------------------

def pack_header(h: Header) -> bytes:
    meta_json = json.dumps(h.meta or {}, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    has_meta = 1 if meta_json else 0
    flags = (h.flags & 0xFF) | ((has_meta & 1) << 1)

    # Fixed portion without crc
    fixed = struct.pack(
        "<4sHHIIHHBBHI",
        MAGIC, VERSION, 0,  # hdr_len placeholder (u16)
        h.width, h.height,
        h.tile, h.overlap,
        h.colorspace & 0xFF, flags & 0xFF,
        0,  # reserved u16
        len(meta_json),
    )
    parts = [fixed, meta_json]
    parts.append(b"\x00" * _pad4(len(meta_json)))  # align
    # Temporary header to compute CRC; hdr_len still 0 here
    tmp = b"".join(parts)
    # Now patch the real hdr_len (including the crc itself at the end)
    hdr_len = len(tmp) + 4
    fixed = struct.pack(
        "<4sHHIIHHBBHI",
        MAGIC, VERSION, hdr_len,
        h.width, h.height,
        h.tile, h.overlap,
        h.colorspace & 0xFF, flags & 0xFF,
        0,  # reserved
        len(meta_json),
    )
    parts[0] = fixed
    tmp = b"".join(parts)
    crc = _crc32(tmp)
    header = tmp + struct.pack("<I", crc)
    return header

def unpack_header(data: bytes) -> Tuple[Header, int]:
    if len(data) < FIXED_SIZE:  # au moins la partie fixe
        raise ValueError("Header too short")
    if data[:4] != MAGIC:
        raise ValueError("Invalid magic")

    magic, ver, hdr_len, width, height, tile, overlap, colorspace, flags, reserved, meta_len = struct.unpack_from(
        FIXED_FMT, data, 0
    )
    if int(ver) != VERSION:
        raise ValueError(f"Incompatible bitstream version {ver} (expected {VERSION})")
    if len(data) < hdr_len:
        raise ValueError("Incomplete header bytes")
    if hdr_len < FIXED_SIZE + 4:  # fixe + CRC (sans meta)
        raise ValueError("Header length invalid")

    # meta JSON
    meta_off = FIXED_SIZE
    meta_json = data[meta_off:meta_off + meta_len]
    if len(meta_json) != meta_len:
        raise ValueError("Header meta truncated")

    # alignement + position du CRC
    pad = _pad4(meta_len)
    crc_off = meta_off + meta_len + pad
    if crc_off + 4 != hdr_len:
        raise ValueError("Header structure mismatch (crc position)")

    expected_crc = struct.unpack_from("<I", data, crc_off)[0]
    actual_crc = _crc32(data[:crc_off])
    if expected_crc != actual_crc:
        raise ValueError(f"Header CRC mismatch (got {actual_crc:#010x}, expected {expected_crc:#010x})")

    meta = {}
    if meta_len:
        try:
            meta = json.loads(meta_json.decode("utf-8"))
        except Exception as e:
            raise ValueError(f"Invalid meta JSON: {e}")

    h = Header(
        width=int(width),
        height=int(height),
        tile=int(tile),
        overlap=int(overlap),
        colorspace=int(colorspace),
        flags=int(flags),
        meta=meta or None,
    )
    return h, int(hdr_len)


# ----------------------------- pack/unpack tile record --------------------

def pack_tile_record(rec: TileRec) -> bytes:
    payload_len = len(rec.payload)
    head = struct.pack(TILE_HEAD_FMT, rec.tile_id, rec.gen_id, rec.qv_id, rec.seed, rec.rec_flags, rec.payload_fmt, payload_len)
    pad = _pad4(payload_len)
    return head + rec.payload + (b"\x00" * pad)

def unpack_tile_record(data: bytes, offset: int = 0) -> Tuple[TileRec, int]:
    if len(data) < offset + TILE_HEAD_SIZE:
        raise ValueError("Tile record truncated")
    tile_id, gen_id, qv_id, seed, rec_flags, payload_fmt, payload_len = struct.unpack_from(TILE_HEAD_FMT, data, offset)
    offset += TILE_HEAD_SIZE
    end = offset + payload_len
    if len(data) < end:
        raise ValueError("Tile payload truncated")
    payload = data[offset:end]
    pad = _pad4(payload_len)
    next_off = end + pad
    return TileRec(
        tile_id=int(tile_id),
        gen_id=int(gen_id),
        qv_id=int(qv_id),
        seed=int(seed),
        rec_flags=int(rec_flags),
        payload_fmt=int(payload_fmt),
        payload=bytes(payload),
    ), next_off


# ----------------------------- pack/unpack footer -------------------------

def pack_footer(tiles_count: int, body: bytes) -> bytes:
    # global_crc32 is CRC of everything before this field (header + tile records)
    crc = _crc32(body)
    return struct.pack("<III", int(tiles_count), 0, crc)

def unpack_footer(data: bytes) -> Tuple[int, int]:
    if len(data) < FOOTER_SIZE:
        raise ValueError("Footer too short")
    tiles_count, reserved, crc = struct.unpack_from("<III", data, len(data) - FOOTER_SIZE)
    return int(tiles_count), int(crc)

# ----------------------------- writer/reader ------------------------------

def write_bitstream(header: Header, records: Iterable[TileRec]) -> bytes:
    """Assemble header + tile records + footer with CRC."""
    hdr = pack_header(header)
    rec_bytes = bytearray()
    count = 0
    for r in records:
        rec_bytes.extend(pack_tile_record(r))
        count += 1
    body = bytes(hdr) + bytes(rec_bytes)
    return body + pack_footer(count, body)

def read_bitstream(blob: bytes) -> Tuple[Header, List[TileRec]]:
    """Parse bitstream, validate CRCs, return (Header, [TileRec...])."""
    if len(blob) < FOOTER_SIZE + 24:
        raise ValueError("Bitstream too short")

    # Footer validation first
    tiles_count, expected_crc = unpack_footer(blob)
    body = blob[:-FOOTER_SIZE]
    actual_crc = _crc32(body)
    if actual_crc != expected_crc:
        raise ValueError(f"Global CRC mismatch (got {actual_crc:#010x}, expected {expected_crc:#010x})")

    # Header
    header, hdr_len = unpack_header(body)
    # Records
    recs: List[TileRec] = []
    off = hdr_len
    end_of_records = len(body)
    while off < end_of_records:
        rec, off = unpack_tile_record(body, off)
        recs.append(rec)
    if len(recs) != tiles_count:
        raise ValueError(f"Tiles count mismatch (footer={tiles_count}, parsed={len(recs)})")
    return header, recs
