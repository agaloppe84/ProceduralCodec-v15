# tests/flows/test_flow_step4.py
import os
import math
import pytest
import torch


@pytest.mark.parametrize("H,W,tile,overlap", [(64, 64, 32, 8)])
def test_flow_step4_end_to_end(monkeypatch, H, W, tile, overlap):
    """
    Flow Step 4 — test unique "conditions réelles" (CPU-only ok)

    Vérifie :
      1) encode_y en ANS0 (par défaut) produit un .pc15 valide
         - header & records OK, payload_fmt == ANS0, header "ANS0", table_id par défaut
      2) decode_y reconstruit une image non-nulle de forme [1,1,H,W]
      3) déterminisme : deux encodes mêmes cfg/seed → mêmes bytes
      4) chemin RAW (debug) reste fonctionnel (payload_fmt == RAW, decode ok)

    Notes :
      - Pas de dépendance GPU : on force PC15_ALLOW_CPU_TESTS=1
      - La reconstruction Step 4 peut être synthétique côté decode (stripes déterministes).
    """

    # -- Env: force ANS0 (par défaut) + chemins CPU-friendly
    monkeypatch.setenv("PC15_PAYLOAD_FMT", "ANS0")
    monkeypatch.setenv("PC15_ALLOW_CPU_TESTS", "1")
    monkeypatch.setenv("PC15_SCORE_BITS", "table")  # rapide

    # -- Imports API publique
    from pc15codec import CodecConfig, DEFAULT_TABLE_ID, ANS0_FMT, RAW_FMT
    from pc15codec.codec import encode_y, decode_y
    from pc15codec.bitstream import read_stream_v15
    from pc15codec.payload import decode_tile_payload
    from pc15codec.rans import available_tables

    # Sanity tables
    avail = available_tables()
    assert DEFAULT_TABLE_ID in avail

    # -- Image d'entrée synthétique (le contenu importe peu ici)
    y = torch.zeros(1, 1, H, W)
    cfg = CodecConfig(tile=tile, overlap=overlap, seed=1337)

    # =========
    # 1) Encode (ANS0) → bitstream v15
    # =========
    enc = encode_y(y, cfg)
    bs = enc["bitstream"]
    assert isinstance(bs, (bytes, bytearray))
    assert enc["bpp"] > 0.0

    # Inspect header + quelques records
    hdr, recs = read_stream_v15(bs)
    assert hdr["width"] == W and hdr["height"] == H
    assert hdr["tile"] == tile and hdr["overlap"] == overlap
    assert len(recs) >= 1

    # Payloads: ANS0 + table id
    for tid, r in enumerate(recs[: min(4, len(recs))]):
        assert r.payload_fmt == ANS0_FMT
        assert r.payload[:4] == b"ANS0"
        L = r.payload[4]
        table_id = r.payload[5 : 5 + L].decode("ascii")
        assert table_id == DEFAULT_TABLE_ID
        # Round-trip symbolique : decode_tile_payload renvoie la 5-uplet (gen_id,qv_id,seed,flags,offsets)
        fields = decode_tile_payload(r.payload)
        assert isinstance(fields, tuple) and len(fields) == 5
        g, qv, s, flg, offs = fields
        assert isinstance(offs, list)
        # seed dérivée par tuile (cfg.seed + tid mod 2^32)
        assert s == ((cfg.seed + r.tile_id) & 0xFFFFFFFF)

    # =========
    # 2) Decode → reconstruction non nulle
    # =========
    yhat = decode_y(bs, device="cpu")
    assert tuple(yhat.shape) == (1, 1, H, W)
    # Step 4 : la recon ne doit pas être toute noire
    assert float(yhat.abs().sum()) > 0.0
    # valeurs bornées raisonnablement en [-1,1]
    assert float(yhat.max()) <= 1.0001 and float(yhat.min()) >= -1.0001

    # =========
    # 3) Déterminisme bit-exact
    # =========
    enc2 = encode_y(y, cfg)
    assert enc2["bitstream"] == bs

    # =========
    # 4) Chemin RAW (debug)
    # =========
    monkeypatch.setenv("PC15_PAYLOAD_FMT", "RAW")
    enc_raw = encode_y(y, cfg)
    bs_raw = enc_raw["bitstream"]
    hdr2, recs2 = read_stream_v15(bs_raw)
    assert (hdr2["width"], hdr2["height"]) == (W, H)
    # Tous les payloads doivent être RAW
    assert all(r.payload_fmt == RAW_FMT for r in recs2)

    # Decode RAW (ne doit pas lever) → recon non nulle (decode Step 4 synthétique)
    yhat_raw = decode_y(bs_raw, device="cpu")
    assert tuple(yhat_raw.shape) == (1, 1, H, W)
    assert float(yhat_raw.abs().sum()) > 0.0
