# tests/flows/test_flow_step5.py
import os
import pytest
import torch

@pytest.mark.parametrize("H,W,tile,overlap", [(64, 64, 32, 8)])
def test_flow_step5_end_to_end(monkeypatch, H, W, tile, overlap):
    """
    Flow Step 5 — encodage RD "coarse" + ANS0, decode recon non nulle.

    Vérifie :
      1) encode_y (ANS0, bits_mode=table) → bitstream v15 valide
         - header & records OK
         - payload_fmt == ANS0, header "ANS0"
         - table_id par défaut présent
         - decode_tile_payload renvoie (gen_id, qv_id, seed, flags, offsets)
           et seed == record.seed (indépendant de la stratégie de dérivation)
      2) decode_y renvoie une image non-nulle [1,1,H,W]
      3) déterminisme bit-exact (on fige le timestamp)
      4) chemin bits_mode=exact fonctionne aussi (sans exigence d’égalité de bpp)
      5) chemin RAW (debug) reste fonctionnel et décode sans erreur

    Notes :
      - PC15_ALLOW_CPU_TESTS=1 pour les environnements sans GPU.
      - On monkeypatch time.time pour garantir le déterminisme du header.
    """

    # -- Env (coarse, CPU-friendly)
    monkeypatch.setenv("PC15_ALLOW_CPU_TESTS", "1")
    monkeypatch.setenv("PC15_PAYLOAD_FMT", "ANS0")
    monkeypatch.setenv("PC15_SCORE_BITS", "table")

    # -- Figer l'horloge pour la stabilité byte-exacte
    import time as _time
    monkeypatch.setattr(_time, "time", lambda: 1_705_000_000)

    # -- Imports API publique
    from pc15codec import CodecConfig, DEFAULT_TABLE_ID, ANS0_FMT, RAW_FMT
    from pc15codec.codec import encode_y, decode_y
    from pc15codec.bitstream import read_stream_v15
    from pc15codec.payload import decode_tile_payload
    from pc15codec.rans import available_tables

    # Tables dispos
    avail = available_tables()
    assert DEFAULT_TABLE_ID in avail

    # Image d'entrée simple
    y = torch.zeros(1, 1, H, W)
    cfg = CodecConfig(tile=tile, overlap=overlap, seed=1337)

    # =========
    # 1) Encode (ANS0, bits_mode=table) → v15 OK
    # =========
    enc = encode_y(y, cfg)
    bs = enc["bitstream"]
    assert isinstance(bs, (bytes, bytearray))
    assert enc["bpp"] > 0.0

    hdr, recs = read_stream_v15(bs)
    assert hdr["width"] == W and hdr["height"] == H
    assert hdr["tile"] == tile and hdr["overlap"] == overlap
    assert len(recs) >= 1

    # Payloads: ANS0 + table id + payload décodable
    for r in recs[: min(4, len(recs))]:
        assert r.payload_fmt == ANS0_FMT
        assert r.payload[:4] == b"ANS0"
        L = r.payload[4]
        table_id = r.payload[5 : 5 + L].decode("ascii")
        assert table_id == DEFAULT_TABLE_ID
        gen_id, qv_id, s, flags, offs = decode_tile_payload(r.payload)
        assert s == r.seed  # robuste, quel que soit le schéma de dérivation
        assert isinstance(offs, list) and all(isinstance(v, int) for v in offs)
        # offsets step5 "coarse" souvent 3 indices (freq/angle/phase), mais on ne durcit pas
        assert len(offs) <= 16

    # =========
    # 2) Decode → recon non nulle
    # =========
    yhat = decode_y(bs, device="cpu")
    assert tuple(yhat.shape) == (1, 1, H, W)
    assert float(yhat.abs().sum()) > 0.0
    assert float(yhat.max()) <= 1.0001 and float(yhat.min()) >= -1.0001

    # =========
    # 3) Déterminisme bit-exact (horloge figée)
    # =========
    enc2 = encode_y(y, cfg)
    assert enc2["bitstream"] == bs

    # =========
    # 4) bits_mode=exact → encode OK
    # =========
    monkeypatch.setenv("PC15_SCORE_BITS", "exact")
    enc_exact = encode_y(y, cfg)
    bs_exact = enc_exact["bitstream"]
    assert isinstance(bs_exact, (bytes, bytearray))
    # On ne demande pas d'égalité de bpp, juste un flux valide et décodable
    hdr_e, recs_e = read_stream_v15(bs_exact)
    assert (hdr_e["width"], hdr_e["height"]) == (W, H)
    for r in recs_e[: min(2, len(recs_e))]:
        assert r.payload_fmt == ANS0_FMT
        assert r.payload[:4] == b"ANS0"
        _ = decode_tile_payload(r.payload)

    # =========
    # 5) Chemin RAW (debug)
    # =========
    monkeypatch.setenv("PC15_PAYLOAD_FMT", "RAW")
    enc_raw = encode_y(y, cfg)
    bs_raw = enc_raw["bitstream"]
    hdr_r, recs_r = read_stream_v15(bs_raw)
    assert all(r.payload_fmt == RAW_FMT for r in recs_r)
    yhat_raw = decode_y(bs_raw, device="cpu")
    assert tuple(yhat_raw.shape) == (1, 1, H, W)
    assert float(yhat_raw.abs().sum()) > 0.0
