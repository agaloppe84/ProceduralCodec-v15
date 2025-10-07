# packages/pc15codec/src/pc15codec/rans_impl.py
# -----------------------------------------------------------------------------
# rANS v15 - implémentation principale (tables référencées, format ANS0)

from __future__ import annotations
from typing import Dict, List, Tuple, Optional

"""
rANS v15 - table-referenced (ANS0)
==================================

But
----
Compresser une suite de symboles 8-bit (0..255) quasi à l’entropie en utilisant
un ANS 32-bit déterministe et rapide, **sans** embarquer les tables dans le
payload. Les tables (freqs/CDF) sont **référencées** via un `table_id` court,
puis chargées côté décodeur.

Payload layout (little-endian)
------------------------------
[0:4]     b"ANS0"                  # MAGIC
[4]       L                        # u8, longueur de `table_id`
[5:5+L]   table_id (ASCII)         # ex: b"v15_default"
[...:]
          stream bytes (sans tables) :
            [0:4]   u32 nsym       # nombre de symboles
            [4:8]   u32 state      # état final x
            [8:]    renorm chunks  # 16-bit LSB-first produits à l’encodage

API
---
- build_rans_tables(symbols, precision=12)
    -> {"precision":P,"freqs":[256*u16],"cdf":[257*u32]}
- rans_encode(symbols, tables_or_id)
    -> bytes payload:
       * si `tables_or_id` est `str` (table_id) : écrit l’entête ANS0
         (MAGIC + L + table_id) puis le stream (nsym + state + chunks).
       * si `tables_or_id` est un dict tables : renvoie **uniquement**
         le stream (nsym + state + chunks), utile pour des tests bas-niveau.
- rans_decode(payload, tables_loader=None)
    -> List[int] symbols ; lit ANS0, charge les tables via `tables_loader`
       (par défaut `pc15codec.rans.load_table_by_id`), puis décompresse.

Notes
-----
- `precision` bornée à [1..15] (base <= 32768) pour des freqs sur u16.
- Renormalisation encodeur/décodeur par pas de 16 bits (LSB-first).
- Ce module ne gère **pas** le cas "payload sans MAGIC" (migration ANS1) :
  on lève une erreur pour garantir la déterminisme et éviter les ambiguïtés.
"""

__all__ = [
    "MAGIC",
    "build_rans_tables",
    "rans_encode",
    "rans_decode",
]

# Marqueur ANS0 (tables référencées par id)
MAGIC = b"ANS0"


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


# [ML/ENTROPY:WILL_STORE]
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
# Core rANS 32-bit (table-based, sans en-tête ANS0)
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


def _rans_core_encode(symbols: List[int], tables: Dict[str, object]) -> bytes:
    """
    Encode rANS en produisant **uniquement** le flux core :
      stream = u32 nsym | u32 final_state | renorm_chunks (16-bit LSB-first)

    Retourne `bytes` prêtes à être préfixées par l'entête ANS0 si nécessaire.
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
    chunks = bytearray()

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
            chunks.append(x & 0xFF)
            chunks.append((x >> 8) & 0xFF)
            x >>= 16
        # Étape core
        x = ((x // f) << P) + (x % f) + c

    # Préfixe: nsym (u32) puis état final (u32), puis les chunks LSB-first
    nsym = len(symbols)
    buf = bytearray()
    buf.extend((
        nsym & 0xFF, (nsym >> 8) & 0xFF, (nsym >> 16) & 0xFF, (nsym >> 24) & 0xFF
    ))
    buf.extend((
        x & 0xFF, (x >> 8) & 0xFF, (x >> 16) & 0xFF, (x >> 24) & 0xFF
    ))
    buf.extend(chunks)
    return bytes(buf)


def _rans_core_decode(stream: bytes, tables: Dict[str, object]) -> List[int]:
    """
    Décode le flux core :
      stream = u32 nsym | u32 final_state | renorm_chunks (16-bit LSB-first)
    """
    if len(stream) < 8:
        raise ValueError("rANS stream too short")

    nsym = int.from_bytes(stream[0:4], "little")
    x = int.from_bytes(stream[4:8], "little")
    renorm = stream[8:]

    P = int(tables["precision"])
    if not (1 <= P <= 15):
        raise ValueError(f"precision out of range: {P}")
    base = 1 << P

    freqs: List[int] = list(tables["freqs"])
    if sum(freqs) != base:
        raise ValueError("freqs must sum to base (1<<precision)")

    # Inverse map L[r] -> symbol, et CDF implicite
    Lmap = [0] * base
    cdf = [0] * 257
    acc = 0
    for i, f in enumerate(freqs):
        cdf[i] = acc
        acc_next = acc + f
        for r in range(acc, acc_next):
            Lmap[r] = i
        acc = acc_next
    cdf[256] = acc

    res = [0] * nsym
    ptr = len(renorm)  # LIFO : on lit les chunks à rebours (fin -> début)
    mask = (1 << P) - 1
    Tdec = _dec_threshold()  # = 1<<16

    for i in range(nsym):
        r = x & mask
        s = Lmap[r]
        res[i] = s
        f = freqs[s]
        c = cdf[s]
        x = f * (x >> P) + (r - c)

        # Renormalisation décodeur
        while x < Tdec:
            if ptr < 2:
                raise ValueError("rANS stream underflow")
            ptr -= 2
            lo = renorm[ptr]
            hi = renorm[ptr + 1]
            x = (x << 16) | (hi << 8) | lo

    return res


# -----------------------------------------------------------------------------
# Enveloppe ANS0 (header + table_id) et API publique
# -----------------------------------------------------------------------------
# [ML/ENTROPY:WILL_STORE]
def rans_encode(symbols: List[int], tables_or_id, precision: int | None = None) -> bytes:
    """
    Encode une liste de symboles 0..255 avec rANS.

    - Si `tables_or_id` est `str` (table_id) :
        écrit l'entête ANS0 (MAGIC + u8 L + table_id ASCII) puis
        le flux core (u32 nsym | u32 state | chunks), en chargeant
        les tables via `pc15codec.rans.load_table_by_id(table_id)`.
    - Si `tables_or_id` est un dict de tables :
        retourne **uniquement** le flux core (sans entête ANS0).
    """
    # 1) Résolution des tables
    if isinstance(tables_or_id, str):
        table_id = tables_or_id
        from pc15codec.rans import load_table_by_id  # lazy import pour éviter les cycles
        tables = load_table_by_id(table_id)
        # 2) Encodage core
        stream = _rans_core_encode(symbols, tables)
        tid = table_id.encode("ascii")
        if len(tid) > 255:
            raise ValueError("table_id too long (max 255 ASCII bytes)")
        return MAGIC + bytes([len(tid)]) + tid + stream
    else:
        tables = tables_or_id
        return _rans_core_encode(symbols, tables)


# [ML/ENTROPY:WILL_STORE]
def rans_decode(payload: bytes, tables_loader=None) -> List[int]:
    """
    Décode un payload ANS0.
    - Lit `MAGIC`, `L`, `table_id`.
    - Charge les tables via `tables_loader(table_id)` (par défaut `pc15codec.rans.load_table_by_id`).
    - Décode le flux core et retourne la liste des symboles.

    NB: Les payloads **sans** entête ANS0 ne sont **pas** supportés ici.
    """
    if not payload.startswith(MAGIC):
        raise ValueError("rans_decode: unsupported payload (missing ANS0 header)")

    if len(payload) < 5:
        raise ValueError("rans_decode: payload too short")

    L = payload[4]
    if len(payload) < 5 + L:
        raise ValueError("rans_decode: truncated table_id")

    table_id = payload[5:5 + L].decode("ascii")
    stream = payload[5 + L:]

    if tables_loader is None:
        from pc15codec.rans import load_table_by_id
        tables_loader = load_table_by_id

    tables = tables_loader(table_id)
    return _rans_core_decode(stream, tables)


# [ML/ENTROPY:WILL_STORE] - encode/décode entropique, stocke des tables/symboles.

__all__ = ["MAGIC", "build_rans_tables", "rans_encode", "rans_decode"]
