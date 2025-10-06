from __future__ import annotations
import pytest

def test_codec_facade_futureproof():
    """
    Ce test vérifie la façade publique `pc15` (API stable),
    sans dépendre d'impls internes encore en chantier (Step 0).
    """
    import pc15 as pc

    # 1) Les symboles doivent exister sur la façade (même si non câblés côté moteur).
    assert hasattr(pc, "CodecConfig")
    assert hasattr(pc, "encode_y")
    assert hasattr(pc, "decode_y")

    # 2) Tant que le moteur n'est pas branché, on tolère des stubs (None) et on skip.
    if not (callable(pc.encode_y) and callable(pc.decode_y)):
        pytest.skip("encode_y/decode_y non câblés au Step 0 — façade future-proof OK")

    # 3) Si jamais c'est câblé plus tôt, on fait un smoke minimal (sans exécution GPU).
    #    Ici on s'arrête à la signature pour éviter des effets de bord pendant Step 0.
    cfg = pc.CodecConfig()
    assert isinstance(cfg.tile, int) and isinstance(cfg.overlap, int)
