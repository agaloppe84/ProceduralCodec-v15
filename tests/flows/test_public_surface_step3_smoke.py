import os
import torch

def test_public_surface_and_payload_modes(monkeypatch):
    # Imports "docs-like"
    from pc15codec import CodecConfig, ANS0_FMT, RAW_FMT, DEFAULT_TABLE_ID
    from pc15codec.codec import encode_y, decode_y
    from pc15codec.rans import available_tables

    # Config simple
    y = torch.zeros(1, 1, 32, 32)
    cfg = CodecConfig(tile=16, overlap=4, seed=42)

    # ANS0 (par défaut)
    monkeypatch.setenv("PC15_PAYLOAD_FMT", "ANS0")
    enc = encode_y(y, cfg)
    yhat = decode_y(enc["bitstream"], device="cpu")
    assert yhat.shape == y.shape
    assert enc["stats"]["payload_mode"] == "ANS0"
    assert DEFAULT_TABLE_ID in available_tables()

    # RAW (debug)
    monkeypatch.setenv("PC15_PAYLOAD_FMT", "RAW")
    enc_raw = encode_y(y, cfg)
    yhat_raw = decode_y(enc_raw["bitstream"], device="cpu")
    assert yhat_raw.shape == y.shape
    assert enc_raw["stats"]["payload_mode"] == "RAW"

    # Sanity des constantes
    assert isinstance(ANS0_FMT, int) and isinstance(RAW_FMT, int)