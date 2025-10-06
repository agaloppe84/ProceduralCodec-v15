from __future__ import annotations
import importlib
import pytest

def test_payload_futureproof():
    """
    Future-proof: tant que la payload n'est pas livrée au Step 1,
    on n'exige ni encode/decode de payload, ni tables rANS dynamiques.
    Ce test vérifie que l'écosystème 'payload' n'empêche pas la base Step 0.
    """
    # 1) Le core Step 0 doit fonctionner : tables rANS packagées + header v15 + I/O
    from pc15codec.rans import load_table_by_id
    from pc15codec.bitstream import pack_v15, unpack_v15, write_bitstream, read_bitstream

    t = load_table_by_id("v15_default")
    assert "precision_bits" in t

    h = {"width": 8, "height": 8, "tile": 8, "overlap": 0, "flags": 0, "meta": {}}
    bs = pack_v15(h)
    # Pas d'I/O disque obligatoire ici (on teste les helpers existent)
    # On vérifie juste que l'unpack roundtrip est OK :
    assert unpack_v15(bs)["width"] == 8

    # 2) Partie payload : module/fonctions peuvent ne PAS exister au Step 0 → skip
    try:
        payload_mod = importlib.import_module("pc15codec.payload")
    except Exception:
        pytest.skip("pc15codec.payload non disponible au Step 0")
        return

    has_encode = hasattr(payload_mod, "encode_tile_payload")
    has_decode = hasattr(payload_mod, "decode_tile_payload")

    if not (has_encode and has_decode):
        pytest.skip("API payload non exposée au Step 0")

    # 3) Si jamais elles existent déjà, on accepte une non-implémentation
    #    (NotImplementedError/RuntimeError/ValueError) pour rester future-proof.
    with pytest.raises((NotImplementedError, RuntimeError, ValueError, TypeError)):
        payload_mod.encode_tile_payload(b"", table_id="v15_default")
