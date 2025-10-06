import pytest
from pc15codec.symbols import pack_symbols, unpack_symbols


def test_symbols_pack_unpack_roundtrip():
    gen_id, qv_id, seed, flags = 7, 3, 42, 1
    offsets = [-2, 0, 5, 127, -128]
    syms = pack_symbols(gen_id, qv_id, seed, flags, offsets)
    g2, q2, s2, f2, offs2 = unpack_symbols(syms)
    assert (g2, q2, s2, f2, offs2) == (gen_id, qv_id, seed, flags, offsets)


def test_symbols_bounds_checks():
    with pytest.raises(ValueError):
        pack_symbols(0x1_0000, 0, 0, 0, [])  # gen_id 16-bit overflow
    with pytest.raises(ValueError):
        pack_symbols(0, 0x1_0000, 0, 0, [])  # qv_id 16-bit overflow
    with pytest.raises(ValueError):
        pack_symbols(0, 0, -1, 0, [])        # seed < 0
    with pytest.raises(ValueError):
        pack_symbols(0, 0, 0, 0x1_00, [])    # flags 8-bit overflow
    with pytest.raises(ValueError):
        pack_symbols(0, 0, 0, 0, [200])      # offset out of [-128,127]

    # Format inconnu / tronqué
    with pytest.raises(ValueError):
        unpack_symbols([0x00])
    with pytest.raises(ValueError):
        # TAG OK mais offsets tronqués
        unpack_symbols([0xF0, 0, 0, 0, 0, 0, 42, 0, 0, 0, 5, 1, 2])
