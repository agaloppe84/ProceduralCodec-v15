# packages/pc15codec/src/pc15codec/rd.py
from __future__ import annotations

"""
pc15codec.rd - utilitaires RD

- bits_proxy_from_payload(payload) : proxy simple 8*len(payload)
- estimate_bits_from_table(symbols, table) : somme -log2(p) depuis une table gelée
- distortion / rd_score : helpers torch pour D, R, RD

Ces fonctions sont CPU-friendly et ne requièrent pas de GPU.
"""

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import math
import numpy as np
import torch

try:
    from pc15metrics.psnr_ssim import ssim as _ssim
except Exception:
    _ssim = None  # fallback if metrics not installed


MetricName = Literal["ssim", "mse", "mixed"]


@dataclass(frozen=True)
class RDConfig:
    """Configuration minimaliste pour un score RD."""
    lambda_rd: float = 0.02
    metric: MetricName = "ssim"
    alpha: float = 0.7  # pour "mixed"


def bits_proxy_from_payload(payload: bytes) -> float:
    """Estimation proxy des bits d'un payload rANS : 8 * len(payload)."""
    return float(len(payload) * 8)


def estimate_bits_from_table(symbols: Sequence[int], table: dict) -> float:
    """
    Estimation **table-based** des bits de `symbols` :
      R ≈ ∑_t -log2( counts[s]/base ),  base = ∑ counts

    Compatible avec les tables gelées ANS0 (clés usuelles: "counts", "precision"
    ou "alphabet_size"). Si `precision` est absent, on déduit `base` = sum(counts).
    """
    if table is None:
        # worst-case safe: 8 bits / symbole
        return float(8.0 * len(symbols))
    counts = table.get("counts", None)
    if not counts:
        return float(8.0 * len(symbols))
    base = (1 << int(table.get("precision", 0))) if table.get("precision") else sum(int(c) for c in counts)
    out = 0.0
    for s in symbols:
        c = counts[int(s) & 0xFF] if (0 <= int(s) < len(counts)) else 0
        if c <= 0:
            c = 1  # évite -inf, pénalise les symboles “improbables”
        out += -math.log2(float(c) / float(base))
    return float(out)


def distortion(
    y: torch.Tensor,
    yhat: torch.Tensor,
    metric: MetricName = "ssim",
    alpha: float = 0.7,
) -> torch.Tensor:
    """Calcule D sur batch (B,1,H,W) → (B,)."""
    assert y.shape == yhat.shape and y.ndim == 4 and y.shape[1] == 1
    if metric == "mse" or _ssim is None:
        D = torch.mean((y - yhat) ** 2, dim=(1, 2, 3))
        if metric == "mixed":
            D = alpha * D + (1.0 - alpha) * D
        return D
    if metric == "ssim":
        return 1.0 - _ssim(y, yhat)
    if metric == "mixed":
        d_ssim = 1.0 - _ssim(y, yhat)
        d_mse = torch.mean((y - yhat) ** 2, dim=(1, 2, 3))
        return float(alpha) * d_ssim + (1.0 - float(alpha)) * d_mse
    raise ValueError(f"Unknown metric: {metric}")


def rd_score(
    y: torch.Tensor,
    yhat: torch.Tensor,
    *,
    cfg: Optional[RDConfig] = None,
    bits: Optional[torch.Tensor | float] = None,
    payload: Optional[bytes] = None,
) -> dict:
    """
    RD = D + λ * R

    - Si `bits` est fourni → R=bits (broadcast si scalaire).
    - Sinon si `payload` est fourni → R = 8*len(payload) (proxy).
    - Sinon → R=1.0 (proxy minimal).

    Retourne un dict: {"D": (B,), "R": (B,), "RD": (B,)}.
    """
    cfg = cfg or RDConfig()
    B = y.shape[0]
    D = distortion(y, yhat, cfg.metric, cfg.alpha)
    if bits is not None:
        R = bits if torch.is_tensor(bits) else torch.tensor([bits] * B, device=y.device, dtype=y.dtype)
        R = R.to(device=y.device, dtype=y.dtype)
        if R.ndim == 0:
            R = R.expand(B)
    elif payload is not None:
        b = bits_proxy_from_payload(payload)
        R = torch.tensor([b] * B, device=y.device, dtype=y.dtype)
    else:
        R = torch.ones((B,), device=y.device, dtype=y.dtype)
    RD = D + float(cfg.lambda_rd) * R
    return {"D": D, "R": R, "RD": RD}
