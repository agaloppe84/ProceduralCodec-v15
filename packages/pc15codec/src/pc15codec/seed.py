from __future__ import annotations

def tile_seed(global_seed: int, tile_id: int) -> int:
    """Derive a deterministic per-tile seed (SplitMix-like hashing).

    Ensures identical inputs ⇒ identical outputs across runs (idempotence).
    """
    x = (global_seed ^ (tile_id * 0x9E3779B97F4A7C15)) & 0xFFFFFFFFFFFFFFFF
    x ^= (x >> 33); x *= 0xff51afd7ed558ccd
    x ^= (x >> 33); x *= 0xc4ceb9fe1a85ec53
    x ^= (x >> 33)
    return x & 0xFFFFFFFF
