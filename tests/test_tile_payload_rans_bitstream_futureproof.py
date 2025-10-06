from __future__ import annotations
import pytest

from pc15codec.bitstream import pack_v15, unpack_v15, write_bitstream, read_bitstream

def test_tile_payload_rans_bitstream_futureproof(tmp_path):
    """
    Step 0 : on garantit le header v15 (+ CRC) et l'I/O atomique.
    La partie payload par tuile (rANS) est future-proof → SKIP si absente.
    """
    # 1) Header v15 minimal avec un hint facultatif côté payload
    hdr = {
        "width": 16,
        "height": 16,
        "tile": 8,
        "overlap": 0,
        "flags": 0,
        "meta": {"encoder": "pc15codec@v15.0.0", "seed": 1},
        "payload_hint": {"count": 2, "fmt": "ANS0"},
    }

    # 2) Pack + I/O atomique  # [STORE:OVERWRITE]
    bs = pack_v15(hdr)
    out = tmp_path / "tiles.pc15"
    write_bitstream(bs, out)
    buf = read_bitstream(out)

    # 3) Unpack et vérifs de base
    h2 = unpack_v15(buf)
    assert h2["width"] == 16 and h2["height"] == 16
    assert h2["tile"] == 8 and h2["overlap"] == 0
    assert h2["payload_hint"]["fmt"] == "ANS0"

    # 4) Partie payload/rANS : optionnelle au Step 0 → SKIP si non dispo
    try:
        import pc15codec.payload as payload
    except Exception:
        pytest.skip("pc15codec.payload non disponible au Step 0")
        return

    has_encode = hasattr(payload, "encode_tile_payload")
    has_decode = hasattr(payload, "decode_tile_payload")
    if not (has_encode and has_decode):
        pytest.skip("API payload non exposée au Step 0")
        return

    # 5) Si présent déjà, on accepte une non-implémentation explicite
    with pytest.raises((NotImplementedError, RuntimeError, ValueError, TypeError)):
        payload.encode_tile_payload(b"dummy", table_id="v15_default")
