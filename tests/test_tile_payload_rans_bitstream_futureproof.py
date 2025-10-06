from pc15codec.bitstream import Header, TileRec, write_bitstream, read_bitstream
from pc15codec.payload import encode_tile_payload, decode_tile_payload


def test_tile_payload_rans_bitstream_roundtrip_futureproof():
    hdr = Header(32, 32, tile=16, overlap=8, colorspace=0, flags=0, meta={"t":"e2e"})
    payload = encode_tile_payload(gen_id=7, qv_id=3, seed=42, flags=1, offsets=[-2,0,5], precision=12)
    rec = TileRec(tile_id=0, gen_id=7, qv_id=3, seed=42, rec_flags=1, payload_fmt=0, payload=payload)
    bs = write_bitstream(hdr, [rec])
    hdr2, recs = read_bitstream(bs)
    g, q, s, f, offs = decode_tile_payload(recs[0].payload)
    assert (g, q, s, f, offs) == (7, 3, 42, 1, [-2, 0, 5])
