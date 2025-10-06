from __future__ import annotations
from pc15codec.seed import tile_seed

def test_seed_determinism():
    s1 = [tile_seed(1234, i) for i in range(16)]
    s2 = [tile_seed(1234, i) for i in range(16)]
    assert s1 == s2 and len(set(s1)) == len(s1)
