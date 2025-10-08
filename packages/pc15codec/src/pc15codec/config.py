# packages/pc15codec/src/pc15codec/config.py
from __future__ import annotations
from dataclasses import dataclass

__all__ = ["CodecConfig"]


@dataclass(frozen=True, slots=True)
class CodecConfig:
    """
    Configuration **publique et stable** du codec PC15 v15 (Step 3 - Y-only).

    Cette config est consommée par `pc15codec.codec.encode_y/decode_y`.
    Elle couvre :
      - le **tiling** (taille/overlap),
      - quelques **flags** de framing,
      - les paramètres **RD** (utiles aux steps suivants),
      - le **seed** de déterminisme (dérivé par tuile à l'encodage),
      - l'identifiant de **tables rANS** gelées.

    Champs
    ------
    tile : int, default=256
        Taille (carrée) des tuiles en pixels. Doit être > 0.
    overlap : int, default=24
        Overlap en pixels (blend inter-tuiles). Doit vérifier 0 <= overlap < tile.
    colorspace : int, default=0
        Espace couleur. 0 = Y-only (Step 3). Chroma viendra plus tard.
    flags : int, default=0
        Champs de bits réservés pour le framing/compat.
    payload_precision : int, default=12
        Précision des tables *si* génération à la volée (u16). Conservé pour la méta.
    lambda_rd : float, default=0.015
        Poids du coût bits (λ) dans l’objectif RD (steps suivants).
    alpha_mix : float, default=0.7
        Mélange SSIM/MSE dans la distortion (steps suivants).
    seed : int, default=1234
        Graine globale. La graine par tuile est dérivée de celle-ci pour garantir
        l’idempotence (mêmes cfg ⇒ mêmes bytes).
    rans_table_id : str, default="v15_default"
        Identifiant de tables rANS gelées. Doit correspondre à un JSON chargeable via
        `pc15codec.rans.load_table_by_id(...)`. Peut être surchargé via l’ENV
        `PC15_RANS_TABLES` pour pointer un répertoire custom.
    ai : object | None, default=None
        Réservé pour des aides “learning-based” futures.  # [ML]

    Notes
    -----
    - La dataclass est **immuable** (`frozen=True`) pour faciliter le hashing et
      l’idempotence des runs.
    - Aucune conversion n’est appliquée : les validations lèvent une `ValueError`
      si les bornes sont violées.
    """

    # Tiling / framing
    tile: int = 256
    overlap: int = 24
    colorspace: int = 0
    flags: int = 0

    # Meta / encodage
    payload_precision: int = 12

    # RD (utilisé aux steps suivants)
    lambda_rd: float = 0.015
    alpha_mix: float = 0.7

    # Déterminisme
    seed: int = 1234

    # rANS
    rans_table_id: str = "v15_default"

    # Réservé (ML)
    ai: object | None = None  # [ML]

    def __post_init__(self) -> None:
        # Validations légères (lèvent ValueError en cas d’inputs invalides)
        if self.tile <= 0:
            raise ValueError("CodecConfig.tile must be > 0")
        if not (0 <= self.overlap < self.tile):
            raise ValueError("CodecConfig.overlap must satisfy 0 <= overlap < tile")
        if not (1 <= int(self.payload_precision) <= 15):
            raise ValueError("CodecConfig.payload_precision must be in [1..15] (u16 freqs)")
        if not isinstance(self.rans_table_id, str) or not self.rans_table_id:
            raise ValueError("CodecConfig.rans_table_id must be a non-empty string")
