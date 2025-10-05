from pc15codec.rans import build_rans_tables

def test_freqs_sum_equals_base_and_shapes():
    syms = [0, 1, 1, 2, 2, 2, 255]
    tables = build_rans_tables(syms, precision=12)
    assert tables["precision"] == 12
    assert len(tables["freqs"]) == 256
    assert len(tables["cdf"]) == 257
    assert sum(tables["freqs"]) == (1 << 12)
