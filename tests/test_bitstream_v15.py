
from pc15codec.bitstream import Header, TileRec, write_bitstream, read_bitstream

def test_bitstream_v15_roundtrip():
    hdr = Header(width=64, height=32, tile=16, overlap=8, colorspace=0, flags=0,
                 meta={"encoder":"pc15codec@v15.0.0","seed":1234})
    recs = [
        TileRec(tile_id=0, gen_id=7, qv_id=3, seed=42, rec_flags=0, payload_fmt=0, payload=b"ANS0\x01\x02"),
        TileRec(tile_id=1, gen_id=7, qv_id=3, seed=43, rec_flags=1, payload_fmt=0, payload=b"ANS0\x03\x04"),
    ]
    bs = write_bitstream(hdr, recs)
    h2, r2 = read_bitstream(bs)
    assert h2.width == 64 and h2.height == 32 and h2.tile == 16 and h2.overlap == 8
    assert len(r2) == 2
    assert r2[0].payload.startswith(b"ANS0")
    assert r2[1].rec_flags == 1
