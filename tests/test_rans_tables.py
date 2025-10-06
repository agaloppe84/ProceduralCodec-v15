from __future__ import annotations
from pc15codec.rans import load_table_by_id
from pc15codec.paths import PathsConfig

def test_rans_packaged_load():
    t = load_table_by_id("v15_default", paths=PathsConfig())
    assert "precision_bits" in t and t["precision_bits"] == 8
