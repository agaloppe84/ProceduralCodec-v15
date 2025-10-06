from __future__ import annotations
import json
import os
import pytest

from pc15codec.rans import load_table_by_id

def test_rans_tables_consistency_packaged():
    """La table packagée doit être auto-cohérente (Step 0)."""
    t = load_table_by_id("v15_default")
    assert t["precision_bits"] == 8
    counts = t["counts"]
    assert isinstance(counts, list) and len(counts) == 256
    assert sum(counts) == (1 << t["precision_bits"])

def test_rans_tables_env_override(monkeypatch, tmp_path):
    """L'ENV PC15_MODELS_DIR doit surcharger la table packagée."""
    root = tmp_path / "models"
    (root / "rans").mkdir(parents=True, exist_ok=True)
    override = {
        "id": "custom_override",
        "version": "15.0.0",
        "precision_bits": 8,
        "alphabet": "byte_256_uniform",
        "alphabet_size": 256,
        "counts": [1] * 256,
    }
    # On remplace l'ID de la table par la même clé 'v15_default' pour
    # vérifier que le loader préfère l'ENV au package.
    (root / "rans" / "v15_default.json").write_text(json.dumps(override))

    monkeypatch.setenv("PC15_MODELS_DIR", str(root))
    t = load_table_by_id("v15_default")
    assert t["id"] == "custom_override"      # preuve qu'on a bien pris l'override
    assert sum(t["counts"]) == (1 << t["precision_bits"])

@pytest.mark.skip(reason="Step 0: pas de build_rans_tables ; sera testé au Step 1")
def test_build_rans_tables_future():
    pass
