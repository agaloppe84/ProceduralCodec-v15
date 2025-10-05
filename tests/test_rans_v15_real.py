import random
import pytest

from pc15codec.rans import MAGIC, build_rans_tables, rans_encode, rans_decode


def test_rans_roundtrip_small():
    syms = [1, 2, 3, 3, 2, 1, 255, 0, 128]
    tables = build_rans_tables(syms, precision=12)
    blob = rans_encode(syms, tables)
    back = rans_decode(blob, None)  # tables ignorées car embarquées
    assert back == syms


def test_rans_roundtrip_random_reproducible():
    random.seed(1234)
    syms = [random.randrange(0, 256) for _ in range(1500)]
    tables = build_rans_tables(syms, precision=12)
    blob = rans_encode(syms, tables)
    back = rans_decode(blob, None)
    assert back == syms


def test_rans_empty_sequence_ok():
    # tables issues d'une distribution quelconque
    base_syms = [1, 1, 2, 2, 2, 3, 255]
    tables = build_rans_tables(base_syms, precision=12)
    blob = rans_encode([], tables)
    back = rans_decode(blob, None)
    assert back == []


def test_build_tables_precision_bounds():
    # bornes OK
    t1 = build_rans_tables([0, 1, 2], precision=1)
    t2 = build_rans_tables([0, 1, 2], precision=15)
    assert sum(t1["freqs"]) == (1 << t1["precision"])
    assert sum(t2["freqs"]) == (1 << t2["precision"])

    # hors bornes → ValueError
    with pytest.raises(ValueError):
        build_rans_tables([0, 1], precision=0)
    with pytest.raises(ValueError):
        build_rans_tables([0, 1], precision=16)


def test_passthrough_when_no_magic():
    raw = bytes([10, 20, 30, 40])
    out = rans_decode(raw, None)
    assert out == list(raw)  # passthrough attendu


def test_decode_too_short_payload_raises():
    # "ANS1" + P (12) + (très) tronqué → trop court
    bad = MAGIC + bytes([12]) + b"\x00" * 10
    with pytest.raises(ValueError):
        rans_decode(bad, None)


def test_zero_frequency_symbol_raises_on_encode():
    # tables construites sans le symbole 99
    syms = [1, 2, 3, 3, 2, 1]
    tables = build_rans_tables(syms, precision=12)
    with pytest.raises(ValueError):
        rans_encode(syms + [99], tables)  # 99 absent → freq=0 → ValueError
