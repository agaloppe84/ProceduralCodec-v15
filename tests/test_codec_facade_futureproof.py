import torch
from pc15codec.codec import CodecConfig, encode_y, decode_y
from pc15codec.bitstream import read_bitstream
from pc15codec.rans import MAGIC
from pc15codec.payload import decode_tile_payload


def _tile_grid_like(H: int, W: int, tile: int, overlap: int):
    assert tile > 0 and 0 <= overlap < tile
    stride = tile - overlap
    ys = list(range(0, max(1, H - overlap), stride))
    xs = list(range(0, max(1, W - overlap), stride))
    rects = []
    for y in ys:
        for x in xs:
            y1 = min(y + tile, H)
            x1 = min(x + tile, W)
            rects.append((y, y1, x, x1))
    return rects


def test_codec_encode_y_produces_valid_bitstream_and_payloads_futureproof():
    H, W = 64, 64
    img = torch.zeros((1, 1, H, W), dtype=torch.float32)  # CPU-friendly
    cfg = CodecConfig(tile=16, overlap=8, payload_precision=12, seed=1234)

    out = encode_y(img, cfg)
    bs = out["bitstream"]
    header, recs = read_bitstream(bs)

    assert header.width == W and header.height == H
    rects = _tile_grid_like(H, W, cfg.tile, cfg.overlap)
    assert len(recs) == len(rects) == out["stats"]["tiles"]

    # Payload invariants (compatible S1→S2+)
    for tid, rec in enumerate(recs[:4]):
        assert rec.payload_fmt == 0
        p = bytes(rec.payload)
        assert p.startswith(MAGIC)
        g, q, s, f, offs = decode_tile_payload(p)

        # Invariants (bornes/types), pas de valeur "stub" imposée
        assert 0 <= g <= 0xFFFF
        assert 0 <= q <= 0xFFFF
        assert 0 <= f <= 0xFF
        assert 0 <= s <= 0xFFFFFFFF
        assert len(offs) <= 255
        assert all(-128 <= o <= 127 for o in offs)


def test_codec_decode_y_shape_dtype_and_range_futureproof():
    H, W = 48, 40
    img = torch.zeros((1, 1, H, W), dtype=torch.float32)
    cfg = CodecConfig(tile=16, overlap=8, payload_precision=12, seed=7)

    enc = encode_y(img, cfg)
    y = decode_y(enc["bitstream"], device="cpu")  # S1: canvas noir; S2+: synthèse

    assert y.shape == (1, 1, H, W)
    assert y.dtype == torch.float32
    assert torch.isfinite(y).all()
    # Contrat amplitude PC15
    assert float(y.min()) >= -1.0 and float(y.max()) <= 1.0
