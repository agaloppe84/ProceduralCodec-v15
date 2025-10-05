from pc15proc.register_all import register_all
from pc15proc.registry import list_generators, get

def test_registry_roundtrip():
    names = register_all(verbose=False)
    infos = list_generators()
    assert len(infos) >= 20, "Expected at least 20 generators discovered"
    all_names = [i.name for i in infos]
    # registry uniqueness
    assert len(set(all_names)) == len(all_names)
    # Some canonical names should be present (e.g., STRIPES)
    assert any(i == "STRIPES" for i in all_names), "STRIPES not registered"
    # get() should work
    g = get("STRIPES")
    assert g.info.name == "STRIPES"
