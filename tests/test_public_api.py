from __future__ import annotations

def test_import_public_api():
    import pc15 as pc
    assert hasattr(pc, "CodecConfig")
    assert hasattr(pc, "read_bitstream")
    assert hasattr(pc, "write_bitstream")
