from __future__ import annotations
import pytest

from pc15codec.rans import load_table_by_id

def test_rans_v15_table_invariants():
    """
    Invariants de base sur la table rANS packagée (Step 0) :
    - somme des comptes = 2^precision_bits
    - alphabet_size cohérent
    - tous les comptes > 0
    """
    t = load_table_by_id("v15_default")
    assert t["precision_bits"] == 8
    assert t["alphabet_size"] == 256

    counts = t["counts"]
    assert isinstance(counts, list) and len(counts) == t["alphabet_size"]
    assert all(isinstance(c, int) and c > 0 for c in counts)
    assert sum(counts) == (1 << t["precision_bits"])

def test_rans_v15_optional_encode_decode():
    """
    Future-proof : si une implémentation rANS (encode/decode) est disponible,
    on fait un round-trip simple ; sinon on SKIP proprement (Step 0).
    """
    try:
        # Ces symboles ne sont pas requis au Step 0
        from pc15codec.rans import rans_encode, rans_decode  # type: ignore
    except Exception:
        pytest.skip("rANS encode/decode non exposés au Step 0")
        return

    data = bytes(range(64))  # payload petit pour test
    enc = rans_encode(data, table_id="v15_default")
    dec = rans_decode(enc, table_id="v15_default")
    assert dec == data
