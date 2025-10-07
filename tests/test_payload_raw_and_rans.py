# tests/test_payload_raw_and_rans.py
from __future__ import annotations
import pytest

from pc15codec.payload import (
    RAW_FMT, ANS0_FMT,
    encode_tile_payload_raw, decode_tile_payload_raw,
    encode_tile_payload, decode_tile_payload,
)

def test_payload_raw_roundtrip():
    data = b"\x00\x01raw\xff"
    fmt, blob = encode_tile_payload_raw(data)
    assert fmt == RAW_FMT == 1
    out = decode_tile_payload_raw(fmt, blob)
    assert out == data

def test_payload_raw_wrong_fmt_raises():
    with pytest.raises(ValueError):
        # mauvais fmt (ANS0) pour le décodeur RAW
        decode_tile_payload_raw(ANS0_FMT, b"anything")

def test_payload_rans_roundtrip_small():
    # petit jeu de symboles pour un test rapide
    gen_id, qv_id, seed, flags = 7, 3, 42, 5
    offsets = [0, 1, 2, 3, 4]

    # précision réduite pour accélérer le build des tables dans le test
    blob = encode_tile_payload(gen_id, qv_id, seed, flags, offsets, precision=8)
    assert isinstance(blob, (bytes, bytearray))
    # On ne fige pas le magic "ANS0" ici pour rester compatible avec d’éventuelles évolutions
    g, q, s, f, off = decode_tile_payload(blob)
    assert (g, q, s, f) == (gen_id, qv_id, seed, flags)
    assert off == offsets
