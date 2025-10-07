# tests/flow/test_ans0_roundtrip.py
import pytest
import torch

from pc15codec.codec import encode_y, decode_y, CodecConfig
from pc15codec.bitstream import read_stream_v15
from pc15codec.payload import ANS0_FMT, RAW_FMT, decode_tile_payload
from pc15codec.rans import DEFAULT_TABLE_ID


def _extract_table_id(payload: bytes) -> str:
    """Lit l'entête ANS0 et extrait le table_id ASCII."""
    assert payload[:4] == b"ANS0"
    L = payload[4]
    return payload[5:5+L].decode("ascii")


@pytest.mark.parametrize("H,W,tile,overlap", [(64, 64, 32, 8)])
def test_ans0_roundtrip_symbolic(monkeypatch, H, W, tile, overlap):
    """
    Step 3 — Round-trip symbolique :
      - encode_y en ANS0 (par défaut)
      - read_stream_v15 pour inspecter les records
      - vérifie payload_fmt == ANS0_FMT et header b"ANS0"
      - vérifie table_id == DEFAULT_TABLE_ID
      - decode_tile_payload → (gen_id,qv_id,seed,flags,offsets)
      - seed == cfg.seed + tid (mod 2^32)
      - decode_y ne plante pas et renvoie un tenseur noir [1,1,H,W]
    """
    monkeypatch.setenv("PC15_PAYLOAD_FMT", "ANS0")  # explicite, même si c'est le défaut

    y = torch.zeros(1, 1, H, W)
    cfg = CodecConfig(tile=tile, overlap=overlap, seed=1337)

    enc = encode_y(y, cfg)
    bitstream = enc["bitstream"]

    # Décode global (ne doit pas lever) et forme attendue
    dec = decode_y(bitstream)
    assert tuple(dec.shape) == (1, 1, H, W)
    assert float(dec.sum()) == 0.0

    # Inspection des records
    header, records = read_stream_v15(bitstream)
    assert len(records) > 0

    for tid, r in enumerate(records):
        assert r.payload_fmt == ANS0_FMT
        assert r.payload.startswith(b"ANS0")

        # table_id dans l'entête ANS0
        table_id = _extract_table_id(r.payload)
        assert table_id == DEFAULT_TABLE_ID

        # round-trip symbolique
        g, qv, seed, flags, offsets = decode_tile_payload(r.payload)
        assert isinstance(g, int) and isinstance(qv, int)
        assert isinstance(seed, int) and isinstance(flags, int)
        assert isinstance(offsets, list)

        expected_seed = (int(cfg.seed) + tid) & 0xFFFFFFFF
        assert seed == expected_seed


def test_raw_mode_debug(monkeypatch):
    """
    Vérifie le fallback RAW (debug) :
      - force PC15_PAYLOAD_FMT=RAW
      - encode_y → tous les payloads en RAW
      - decode_y ne plante pas
    """
    monkeypatch.setenv("PC15_PAYLOAD_FMT", "RAW")

    y = torch.zeros(1, 1, 32, 32)
    cfg = CodecConfig(tile=16, overlap=4, seed=1)

    enc = encode_y(y, cfg)
    header, records = read_stream_v15(enc["bitstream"])
    assert len(records) > 0
    assert all(r.payload_fmt != ANS0_FMT and r.payload_fmt == RAW_FMT for r in records)

    dec = decode_y(enc["bitstream"])
    assert tuple(dec.shape) == (1, 1, 32, 32)
