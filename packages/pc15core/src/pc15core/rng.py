from __future__ import annotations
import torch

_MASK64 = 0xFFFFFFFFFFFFFFFF
_SIGNBIT = 1 << 63
_MOD64 = 1 << 64

def splitmix64(x: int) -> int:
    x &= _MASK64
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    z = x
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & _MASK64
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & _MASK64
    z ^= z >> 31
    return z & _MASK64

def derive_seed64(*keys: int) -> int:
    s = 0x1234ABCD9876EF01
    for k in keys:
        s = splitmix64(s ^ (k & _MASK64))
    return s  # unsigned 64-bit range

def _to_int64_signed(u: int) -> int:
    return u - _MOD64 if (u & _SIGNBIT) else u

# ✨ NOUVEAU: helper public
def to_int64_signed(u: int) -> int:
    """Public wrapper: unsigned 64-bit -> signed int64 (two's complement)."""
    return _to_int64_signed(u)

def seed_tensor(keys: torch.Tensor) -> torch.Tensor:
    out = keys.clone().to(dtype=torch.int64)
    flat = out.view(-1)
    for i in range(flat.numel()):
        u = derive_seed64(int(flat[i].item()))
        flat[i] = _to_int64_signed(u)
    return out

def make_generators(seeds: torch.Tensor) -> list[torch.Generator]:
    gens: list[torch.Generator] = []
    for s in seeds.view(-1).tolist():
        g = torch.Generator(device=seeds.device.type)
        g.manual_seed(_to_int64_signed(int(s) & _MASK64))
        gens.append(g)
    return gens
