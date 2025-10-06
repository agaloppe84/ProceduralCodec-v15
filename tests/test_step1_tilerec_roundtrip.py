from __future__ import annotations
from pc15codec.bitstream import TileRec

def test_step1_tilerec_roundtrip():
    r = TileRec(tile_id=7, gen_id=3, qv_id=1, seed=42, rec_flags=5, payload_fmt=0, payload=b"ANS0\x00\x01")
    d = r.to_dict()
    r2 = TileRec.from_dict(d)
    assert r2 == r
    assert r2.payload.startswith(b"ANS0")
