from __future__ import annotations
from typing import List, Tuple

TAG_V1 = 0xF0  # tag de version pour évolution future

def pack_symbols(gen_id: int, qv_id: int, seed: int, flags: int, offsets: List[int]) -> List[int]:
    if not (0 <= gen_id <= 0xFFFF and 0 <= qv_id <= 0xFFFF):
        raise ValueError("gen_id/qv_id out of range")
    if not (0 <= seed <= 0xFFFFFFFF):
        raise ValueError("seed out of range")
    if not (0 <= flags <= 0xFF):
        raise ValueError("flags out of range")
    if len(offsets) > 255:
        raise ValueError("too many offsets (max=255)")
    syms: List[int] = [
        TAG_V1,
        gen_id & 0xFF, (gen_id >> 8) & 0xFF,
        qv_id  & 0xFF, (qv_id  >> 8) & 0xFF,
        flags & 0xFF,
        seed & 0xFF, (seed >> 8) & 0xFF, (seed >> 16) & 0xFF, (seed >> 24) & 0xFF,
        len(offsets) & 0xFF,
    ]
    for o in offsets:
        if not (-128 <= int(o) <= 127):
            raise ValueError("offset out of range [-128,127]")
        syms.append((int(o) + 128) & 0xFF)  # signé → non signé
    return syms

def unpack_symbols(syms: List[int]) -> Tuple[int,int,int,int,List[int]]:
    if not syms or syms[0] != TAG_V1:
        raise ValueError("unknown symbol stream format")
    gen_id = syms[1] | (syms[2] << 8)
    qv_id  = syms[3] | (syms[4] << 8)
    flags  = syms[5]
    seed   = syms[6] | (syms[7] << 8) | (syms[8] << 16) | (syms[9] << 24)
    n      = syms[10]
    offs = [(b - 128) for b in syms[11:11+n]]
    if len(offs) != n:
        raise ValueError("truncated offsets")
    return gen_id, qv_id, seed, flags, offs
