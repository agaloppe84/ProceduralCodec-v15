from __future__ import annotations
from pc15codec.bitstream import pack_v15, unpack_v15

def test_header_roundtrip_crc():
    h = dict(W=640, H=480, tile=256, overlap=24,
             rans_id="v15_default", gens=["STRIPES"], qv=[], flags=0)
    b = pack_v15(h)
    h2 = unpack_v15(b)
    assert h2 == h
