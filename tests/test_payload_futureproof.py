from pc15codec.payload import encode_tile_payload, decode_tile_payload
from pc15codec.rans import MAGIC


def test_payload_encode_decode_roundtrip():
    payload = encode_tile_payload(
        gen_id=7, qv_id=3, seed=42, flags=1, offsets=[-2, 0, 5], precision=12
    )
    assert payload.startswith(MAGIC)
    g, q, s, f, offs = decode_tile_payload(payload)
    assert (g, q, s, f, offs) == (7, 3, 42, 1, [-2, 0, 5])
