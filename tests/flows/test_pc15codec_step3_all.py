# tests/flows/test_pc15codec_step3_all.py
import os
import pytest
import torch

from pc15codec import CodecConfig, ANS0_FMT, RAW_FMT, DEFAULT_TABLE_ID
from pc15codec.codec import encode_y, decode_y
from pc15codec.bitstream import read_stream_v15
from pc15codec.payload import decode_tile_payload, encode_tile_payload
from pc15codec.rans import available_tables, load_table_by_id


def _extract_table_id(payload: bytes) -> str:
    """Lit l'entête ANS0 et extrait le table_id ASCII."""
    assert payload[:4] == b"ANS0", "payload must start with ANS0"
    L = payload[4]
    return payload[5:5 + L].decode("ascii")


def test_codecconfig_import_surfaces_identity():
    """La même dataclass doit être exposée partout (package, config, codec)."""
    from pc15codec import CodecConfig as C_pkg
    from pc15codec.config import CodecConfig as C_cfg
    from pc15codec.codec import CodecConfig as C_codec
    assert C_pkg is C_cfg is C_codec


@pytest.mark.parametrize("H,W,tile,overlap", [(64, 64, 32, 8)])
def test_ans0_roundtrip_end_to_end(monkeypatch, H, W, tile, overlap):
    """
    Step 3 — Round-trip symbolique ANS0 (codec end-to-end) + inspection bitstream.
    """
    monkeypatch.setenv("PC15_ALLOW_CPU_TESTS", "1")
    monkeypatch.setenv("PC15_PAYLOAD_FMT", "ANS0")  # explicite, défaut attendu

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


def test_raw_mode_debug_path(monkeypatch):
    """
    Fallback RAW (debug) : force ENV, records en RAW, decode OK.
    """
    monkeypatch.setenv("PC15_ALLOW_CPU_TESTS", "1")
    monkeypatch.setenv("PC15_PAYLOAD_FMT", "RAW")

    y = torch.zeros(1, 1, 32, 32)
    cfg = CodecConfig(tile=16, overlap=4, seed=7)

    enc = encode_y(y, cfg)
    header, records = read_stream_v15(enc["bitstream"])

    assert len(records) > 0
    assert all(r.payload_fmt == RAW_FMT and r.payload_fmt != ANS0_FMT for r in records)

    dec = decode_y(enc["bitstream"])
    assert tuple(dec.shape) == (1, 1, 32, 32)


def test_tables_available_and_loader(monkeypatch, tmp_path):
    """
    Vérifie:
      - available_tables() expose au moins DEFAULT_TABLE_ID,
      - load_table_by_id(...) renvoie un dict canonique {"precision","freqs","cdf"},
      - override via PC15_RANS_TABLES avec un JSON minimal (uniforme) fonctionne,
      - encode/decode ANS0 avec table custom (payload API) fonctionne.
    """
    monkeypatch.setenv("PC15_ALLOW_CPU_TESTS", "1")

    # 1) Tables packagées : DEFAULT_TABLE_ID doit exister
    avail = available_tables()
    assert isinstance(avail, list) and len(avail) > 0
    assert DEFAULT_TABLE_ID in avail

    tbl = load_table_by_id(DEFAULT_TABLE_ID)
    assert {"precision", "freqs", "cdf"} <= set(tbl.keys())
    assert len(tbl["freqs"]) == 256 and len(tbl["cdf"]) == 257
    assert tbl["cdf"][256] == sum(tbl["freqs"])

    # 2) Override via PC15_RANS_TABLES avec une table minimale
    custom_root = tmp_path / "rans_tables"
    custom_root.mkdir(parents=True, exist_ok=True)
    custom_id = "custom_uniform"

    # JSON uniforme ultra court (sera coercé en P=8, freqs=[1]*256)
    (custom_root / f"{custom_id}.json").write_text('{"alphabet":"byte_256_uniform"}', encoding="utf-8")

    monkeypatch.setenv("PC15_RANS_TABLES", str(custom_root))

    # La table custom doit apparaître dans la liste et se charger en format canonique
    avail2 = available_tables()
    assert custom_id in avail2

    tbl_custom = load_table_by_id(custom_id)
    assert {"precision", "freqs", "cdf"} <= set(tbl_custom.keys())
    assert len(tbl_custom["freqs"]) == 256 and len(tbl_custom["cdf"]) == 257

    # 3) Round-trip avec l'API payload directe sur la table custom
    g, qv, seed, flags, offs = 7, 3, 42, 1, [0, 1, 2]
    fmt, blob = encode_tile_payload(g, qv, seed, flags, offs, table_id=custom_id)
    assert fmt == ANS0_FMT and blob.startswith(b"ANS0")
    # table_id doit être celui qu'on a demandé
    assert _extract_table_id(blob) == custom_id

    g2, qv2, seed2, flags2, offs2 = decode_tile_payload(blob)
    assert (g2, qv2, seed2, flags2, offs2) == (g, qv, seed, flags, offs)


def test_top_level_exports_present():
    """
    Vérifie que le top-level package expose bien les constantes publiques utiles.
    """
    from pc15codec import ANS0_FMT as A0, RAW_FMT as RW, DEFAULT_TABLE_ID as DID
    assert isinstance(A0, int) and isinstance(RW, int)
    assert isinstance(DID, str) and len(DID) > 0
