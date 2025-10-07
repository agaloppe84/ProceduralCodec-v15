# packages/pc15codec/src/pc15codec/payload.py
# -----------------------------------------------------------------------------
# Payloads de tuiles (PC15 v15) - bi-mode ANS0 (rANS) / RAW (debug)
# [ML/ENTROPY:WILL_STORE] - ce module produit/consomme des blobs destinés au bitstream.

from __future__ import annotations
from typing import Tuple, List

from .symbols import pack_symbols, unpack_symbols
from .rans_impl import rans_encode, rans_decode
from .rans import DEFAULT_TABLE_ID

__all__ = [
    # formats
    "ANS0_FMT", "RAW_FMT",
    # API ANS0 haut-niveau (symboles <-> payload ANS0)
    "encode_tile_payload", "decode_tile_payload",
    # API RAW (debug)
    "encode_tile_payload_raw", "decode_tile_payload_raw",
]

# -----------------------------------------------------------------------------
# Formats de payload (convention PC15 v15)
# -----------------------------------------------------------------------------
#: Format rANS (ANS0) : header b"ANS0" + u8 L + table_id + stream (nsym|state|chunks)
ANS0_FMT: int = 0
#: Format RAW : octets passants (debug / compat)
RAW_FMT: int = 1


# -----------------------------------------------------------------------------
# ANS0 (rANS) — API haut-niveau (symboles <-> payload)
# -----------------------------------------------------------------------------
def encode_tile_payload(
    gen_id: int,
    qv_id: int,
    seed: int,
    flags: int,
    offsets: List[int],
    table_id: str = DEFAULT_TABLE_ID,
) -> Tuple[int, bytes]:
    """
    Encode les **symboles de tuile** en payload **ANS0 (rANS)**.

    Paramètres
    ----------
    gen_id : int
        Identifiant du générateur procédural sélectionné pour la tuile.
    qv_id : int
        Identifiant de la quantification vectorielle (codebook/entrée).
    seed : int
        Graine (seed) déterministe pour la synthèse de la tuile.
    flags : int
        Bits d’options/état pour la tuile.
    offsets : List[int]
        Offsets fins et/ou paramètres additionnels (liste bornée par `pack_symbols`).
    table_id : str, par défaut DEFAULT_TABLE_ID
        Identifiant de tables rANS gelées à utiliser (ex: "v15_default").

    Retour
    ------
    (fmt, payload) : (int, bytes)
        - `fmt == ANS0_FMT` (0)
        - `payload` commence par `b"ANS0"` puis contient `table_id` et le flux rANS.

    Détails
    -------
    1. Les champs `(gen_id, qv_id, seed, flags, offsets)` sont packés en une
       **suite de symboles 0..255** via `pack_symbols(...)`.
    2. La liste de symboles est compressée via rANS avec les **tables référencées**
       par `table_id`. L’en-tête ANS0 est automatiquement préfixé par `rans_encode`.

    Exceptions
    ----------
    TypeError / ValueError si les types/tailles sortent des bornes définies par
    `pack_symbols` ou si `table_id` est invalide.
    """
    if not isinstance(offsets, (list, tuple)):
        raise TypeError("encode_tile_payload: `offsets` must be a list/tuple of ints")
    syms = pack_symbols(gen_id, qv_id, seed, flags, list(map(int, offsets)))
    blob = rans_encode(syms, table_id)  # écrit MAGIC + table_id + stream
    return ANS0_FMT, blob


def decode_tile_payload(payload: bytes):
    """
    Décode un **payload ANS0 (rANS)** vers les symboles de tuile.

    Paramètres
    ----------
    payload : bytes
        Flux produit par `encode_tile_payload(...)` (entête b"ANS0" attendu).

    Retour
    ------
    (gen_id, qv_id, seed, flags, offsets) : Tuple[int, int, int, int, List[int]]

    Détails
    -------
    1. `rans_decode` vérifie l’entête `ANS0`, lit `table_id`, charge les tables,
       puis reconstruit la **liste de symboles 0..255**.
    2. `unpack_symbols` reconstitue la 5-uplet `(gen_id, qv_id, seed, flags, offsets)`.

    Exceptions
    ----------
    ValueError si l’entête est invalide ou si le stream est corrompu.
    """
    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError("decode_tile_payload: `payload` must be bytes")
    syms = rans_decode(bytes(payload))  # lit MAGIC, charge tables, décode stream
    return unpack_symbols(syms)


# -----------------------------------------------------------------------------
# RAW (legacy / debug)
# -----------------------------------------------------------------------------
def encode_tile_payload_raw(data: bytes) -> Tuple[int, bytes]:
    """
    Encode **RAW** → `(fmt, payload)` pour debug/compatibilité.

    Paramètres
    ----------
    data : bytes
        Octets à passer tels quels.

    Retour
    ------
    (fmt, payload) : (int, bytes)
        - `fmt == RAW_FMT` (1)
        - `payload == data` (copie immuable)

    Remarque
    --------
    Ce mode bypass l’entropie et **ne doit pas** être utilisé en production.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("encode_tile_payload_raw: data must be bytes")
    return RAW_FMT, bytes(data)


def decode_tile_payload_raw(fmt: int, payload: bytes) -> bytes:
    """
    Décode `(fmt, payload)` lorsque `fmt == RAW_FMT`.

    Paramètres
    ----------
    fmt : int
        Doit valoir `RAW_FMT` (1).
    payload : bytes
        Octets encodés précédemment en mode RAW.

    Retour
    ------
    bytes
        Les octets d’origine.

    Exceptions
    ----------
    ValueError si `fmt != RAW_FMT`.
    TypeError si `payload` n’est pas un buffer d’octets.
    """
    if fmt != RAW_FMT:
        raise ValueError("decode_tile_payload_raw: wrong fmt (expected RAW_FMT=1)")
    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError("decode_tile_payload_raw: payload must be bytes")
    return bytes(payload)
