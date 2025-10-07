# packages/pc15codec/src/pc15codec/rans_impl.py
# -----------------------------------------------------------------------------
# rANS v15 - implémentation principale (tables embarquées)

# packages/pc15codec/src/pc15codec/rans.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional

"""
rANS v15 - table-based (embedded tables)
========================================

But
----
Compresser une suite de symboles 8-bit (0..255) quasi à l’entropie en utilisant
un ANS 32-bit déterministe et rapide, avec tables (freqs/CDF) **embarquées** dans
le payload. Cela rend chaque payload auto-portant et simple à décoder.

Payload layout (little-endian)
------------------------------
[0:4]    b"ANS1"                 # MAGIC
[4]      P                       # precision (1..15) => base = 1<<P
[5:5+512] 256 * u16 freqs        # fréquences normalisées (somme = base)
[...+512: +4]  u32 nsym          # nombre de symboles
[...: -4]      stream bytes      # chunks 16-bit LSB-first (renormalisation)
[-4:]   u32 final_state          # état final x

API
---
- build_rans_tables(symbols, precision=12) -> {"precision":P,"freqs":[...],"cdf":[...]}
- rans_encode(symbols, tables)  -> bytes payload (MAGIC + tables + stream + state)
- rans_decode(payload, tables=None) -> List[int]  # ignore `tables` si MAGIC présent

Notes
-----
- `precision` bornée à [1..15] pour garder les freqs sur u16 (base <= 32768).
- Les symboles absents du corpus ont freq=0 ; ceux présents ont au moins 1.
- rANS encode en parcourant les symboles **à l'envers** et émet des chunks 16-bit
  LSB-first lors de la renormalisation ; le décodeur lit ces chunks **à rebours (LIFO)**.
- Compat: si le payload **ne** commence **pas** par MAGIC, on renvoie les octets tels quels
  (mode “passthrough” utile en phase de transition).
"""

__all__ = [
    "MAGIC",
    "build_rans_tables",
    "rans_encode",
    "rans_decode",
]

MAGIC = b"ANS1"  # payload marker for table-embedded rANS v1


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _normalize_histogram(hist: List[int], precision: int) -> List[int]:
    """
    Convertit un histogramme en 256 cases en un tableau de fréquences `alloc`
    d'entiers positifs dont la somme vaut `base = 1<<precision`.

    - Garantit que tout symbole **présent** (hist[i] > 0) reçoive au moins 1.
    - Les symboles absents restent à 0.
    - Ajuste par une stratégie "largest remainder" + corrections pour que
      sum(alloc) == base, avec priorité à la stabilité/déterminisme.
    """
    base = 1 << int(precision)
    total = sum(hist)
    if total <= 0:
        out = [0] * 256
        out[0] = base
        return out

    alloc = [0] * 256
    remainders: List[Tuple[float, int]] = []
    s = 0
    for i, h in enumerate(hist):
        if h <= 0:
            continue
        f = (h * base) / total
        q = int(f)  # floor
        if q == 0:
            q = 1  # ensure non-zero for present symbols
        alloc[i] = q
        s += q
        remainders.append((f - q, i))

    # Si on dépasse base, on retire 1 aux plus petites fractions (puis aux plus gros)
    if s > base:
        remainders.sort(key=lambda x: (x[0], hist[x[1]]))  # petites fractions d'abord
        i = 0
        while s > base and i < len(remainders):
            _, sym = remainders[i]
            if alloc[sym] > 1:
                alloc[sym] -= 1
                s -= 1
            i += 1
        if s > base:
            order = sorted(range(256), key=lambda k: alloc[k], reverse=True)
            j = 0
            while s > base and j < 256:
                k = order[j]
                if alloc[k] > 1:
                    alloc[k] -= 1
                    s -= 1
                j += 1

    # Si on est en dessous, on ajoute 1 aux plus grandes fractions (puis aux plus fréquents)
    if s < base:
        remainders.sort(key=lambda x: (-x[0], -hist[x[1]]))  # grandes fractions d'abord
        i = 0
        while s < base and i < len(remainders):
            _, sym = remainders[i]
            alloc[sym] += 1
            s += 1
            i += 1
        if s < base:
            order = sorted(range(256), key=lambda k: hist[k], reverse=True)
            j = 0
            while s < base and j < 256:
                k = order[j]
                alloc[k] += 1
                s += 1
                j += 1

    # garde-fou final
    delta = base - sum(alloc)
    if delta != 0:
        alloc[0] += delta
    return alloc


def build_rans_tables(symbols: List[int], precision: int = 12) -> Dict[str, object]:
    """
    Construit les tables rANS (freqs + CDF) à partir d'un échantillon de symboles.

    precision: 1..15 (u16)
    return: {"precision": P, "freqs": List[int(256)], "cdf": List[int(257)]}
    """
    if not (1 <= int(precision) <= 15):  # u16 freqs => base <= 32768
        raise ValueError(f"precision out of range: {precision}")

    hist = [0] * 256
    for s in symbols:
        hist[int(s) & 0xFF] += 1

    freqs = _normalize_histogram(hist, int(precision))
    base = 1 << int(precision)
    if sum(freqs) != base:
        raise AssertionError("freqs must sum to base (1<<precision)")

    cdf = [0] * 257
    acc = 0
    for i in range(256):
        cdf[i] = acc
        acc += freqs[i]
    cdf[256] = acc  # == base
    return {"precision": int(precision), "freqs": freqs, "cdf": cdf}


# -----------------------------------------------------------------------------
# Core rANS 32-bit (table-based)
# -----------------------------------------------------------------------------
def _enc_threshold(freq: int, precision: int) -> int:
    """
    Seuil encodeur : émettre tant que x >= f << (32 - P)
    (garantit x < 2^32 après l'étape core).
    """
    return int(freq) << (32 - int(precision))


def _dec_threshold() -> int:
    """
    Seuil décodeur : recharger des 16 bits tant que x < (1 << 16).
    """
    return 1 << 16


def rans_encode(symbols: List[int], tables: Dict[str, object]) -> bytes:
    """
    Encode des symboles 0..255 en flux rANS avec tables embarquées (MAGIC + tables + stream + state).

    Payload layout:
      - 4 bytes : b"ANS1"
      - 1 byte  : precision P
      - 256*u16 : freqs (little-endian)
      - u32     : nsym
      - ...     : renorm chunks (16-bit LSB-first, variable length)
      - u32     : final state x (little-endian)
    """
    P = int(tables["precision"])
    if not (1 <= P <= 15):
        raise ValueError(f"precision out of range: {P}")

    freqs: List[int] = list(tables["freqs"])  # 256
    cdf: List[int] = list(tables["cdf"])      # 257
    if len(freqs) != 256 or len(cdf) != 257:
        raise ValueError("invalid tables shape")
    if sum(freqs) != (1 << P):
        raise ValueError("freqs must sum to base (1<<precision)")

    # État initial haut (convention rANS 32-bit avec renorm 16-bit)
    x = 1 << 16
    out = bytearray()

    # Encode en ordre inverse
    for s in reversed(symbols):
        s = int(s) & 0xFF
        f = freqs[s]
        c = cdf[s]
        if f <= 0:
            raise ValueError(f"zero-frequency symbol encountered: {s}")
        # Renormalisation encodeur
        T = _enc_threshold(f, P)  # f << (32 - P)
        while x >= T:
            out.append(x & 0xFF)
            out.append((x >> 8) & 0xFF)
            x >>= 16
        # Étape core
        x = ((x // f) << P) + (x % f) + c

    # état final (little-endian u32)
    state_bytes = bytes((
        x & 0xFF,
        (x >> 8) & 0xFF,
        (x >> 16) & 0xFF,
        (x >> 24) & 0xFF,
    ))

    # header + tables + nsym
    buf = bytearray()
    buf.extend(MAGIC)
    buf.append(P & 0xFF)
    for f in freqs:  # u16 LE
        if not (0 <= f <= 0xFFFF):
            raise ValueError("frequency out of u16 range")
        buf.append(f & 0xFF)
        buf.append((f >> 8) & 0xFF)

    nsym = len(symbols)
    buf.extend((
        nsym & 0xFF, (nsym >> 8) & 0xFF, (nsym >> 16) & 0xFF, (nsym >> 24) & 0xFF
    ))

    # stream + final state
    buf.extend(out)          # renorm chunks (LSB-first 16-bit)
    buf.extend(state_bytes)  # final state
    return bytes(buf)


def rans_decode(data: bytes, tables: Optional[Dict[str, object]] = None) -> List[int]:
    """
    Décode un payload produit par rans_encode. Si `data` ne commence pas par MAGIC,
    on retourne les octets tels quels (mode passthrough).

    NB: si MAGIC présent, les tables embarquées **prennent le pas** sur `tables`.
    """
    if not data.startswith(MAGIC):
        # Legacy / passthrough (utile pendant les migrations)
        return [b for b in data]

    # Longueur minimale: MAGIC(4) + P(1) + freqs(512) + nsym(4) + state(4)
    if len(data) < 4 + 1 + 512 + 4 + 4:
        raise ValueError("rANS payload too short")

    P = data[4]
    if not (1 <= P <= 15):
        raise ValueError(f"invalid precision in payload: {P}")
    base = 1 << P

    # Tables embarquées
    freqs = [int.from_bytes(data[5 + 2 * i: 7 + 2 * i], "little") for i in range(256)]
    if sum(freqs) != base:
        raise ValueError("embedded freqs sum mismatch")
    off = 5 + 512
    nsym = int.from_bytes(data[off:off + 4], "little")
    off += 4
    if nsym < 0:
        raise ValueError("invalid nsym")

    # stream + final state
    if len(data) < off + 4:
        raise ValueError("corrupted payload (missing final state)")
    stream = data[off:-4]
    x = int.from_bytes(data[-4:], "little")

    # Inverse map L[r] -> symbol (utilise la CDF implicite)
    L = [0] * base
    cdf = [0] * 257
    acc = 0
    for i, f in enumerate(freqs):
        cdf[i] = acc
        acc_next = acc + f
        for r in range(acc, acc_next):
            L[r] = i
        acc = acc_next
    cdf[256] = acc

    res = [0] * nsym
    ptr = len(stream)  # LIFO : on lit les chunks à rebours (fin -> début)
    mask = (1 << P) - 1
    Tdec = _dec_threshold()  # = 1<<16

    for i in range(nsym):
        r = x & mask
        s = L[r]
        res[i] = s
        f = freqs[s]
        c = cdf[s]
        x = f * (x >> P) + (r - c)

        # Renormalisation décodeur
        while x < Tdec:
            if ptr < 2:
                raise ValueError("rANS stream underflow")
            ptr -= 2
            lo = stream[ptr]
            hi = stream[ptr + 1]
            x = (x << 16) | (hi << 8) | lo

    return res



# [ML/ENTROPY:WILL_STORE] - encode/décode entropique, stocke des tables/symboles.


__all__ = ["MAGIC", "build_rans_tables", "rans_encode", "rans_decode"]
