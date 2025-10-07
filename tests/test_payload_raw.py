# tests/test_payload_raw.py
from pc15codec.payload import RAW_FMT, ANS0_FMT, encode_tile_payload_raw, decode_tile_payload_raw
import pytest

def test_payload_raw_roundtrip():
    data = b"\x00\x01raw\xff"
    fmt, blob = encode_tile_payload_raw(data)
    assert fmt == RAW_FMT == 1
    out = decode_tile_payload_raw(fmt, blob)
    assert out == data

def test_payload_raw_wrong_fmt_raises():
    with pytest.raises(ValueError):
        decode_tile_payload_raw(ANS0_FMT, b"anything")
