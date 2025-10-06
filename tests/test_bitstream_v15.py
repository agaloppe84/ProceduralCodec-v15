from __future__ import annotations

from pc15codec.bitstream import pack_v15, unpack_v15, write_bitstream, read_bitstream

def test_bitstream_v15_roundtrip(tmp_path):
    # Header v15 minimal (Step 0) : on encode tout ce qu'on veut, c'est du JSON packé + CRC32.
    hdr = {
        "width": 64,
        "height": 32,
        "tile": 16,
        "overlap": 8,
        "colorspace": 0,
        "flags": 0,
        "meta": {"encoder": "pc15codec@v15.0.0", "seed": 1234},
    }

    # 1) Pack header -> bytes
    bs = pack_v15(hdr)

    # 2) Écriture atomique puis relecture  # [STORE:OVERWRITE]
    out = tmp_path / "stream.pc15"
    write_bitstream(bs, out)
    buf = read_bitstream(out)

    # 3) Unpack du header et vérifs
    h2 = unpack_v15(buf)
    assert h2["width"] == 64 and h2["height"] == 32
    assert h2["tile"] == 16 and h2["overlap"] == 8
    assert h2["meta"]["seed"] == 1234
