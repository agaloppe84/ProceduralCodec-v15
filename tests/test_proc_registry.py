from pc15proc.registry import register, get, list_generators
from pc15proc.stripes import GEN

def test_registry_roundtrip():
    register(GEN)
    g = get("STRIPES")
    assert g.info.name == "STRIPES"
    infos = list_generators()
    assert any(i.name == "STRIPES" for i in infos)
