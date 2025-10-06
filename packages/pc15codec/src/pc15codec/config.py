from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class CodecConfig:
    """Stable codec configuration exposed publicly.

    Params
    ------
    tile : int
        Tile size in pixels (square tiles).
    overlap : int
        Overlap size for blending.
    lambda_rd : float
        Lagrange multiplier for rate-distortion objective.
    alpha_mix : float
        Mix between SSIM and MSE (distortion blend).
    rans_table_id : str
        Identifier for the frozen rANS table to use.
    ai : object | None
        Reserved for future learning-based helpers.  # [ML]
    """
    tile: int = 256
    overlap: int = 24
    lambda_rd: float = 0.015
    alpha_mix: float = 0.7
    rans_table_id: str = "v15_default"
    ai: object | None = None  # [ML]

__all__ = ["CodecConfig"]
