# packages/pc15codec/src/pc15codec/rd.py
from __future__ import annotations

"""
pc15codec.rd — squelettes RD (Step 4)

But
---
Fournir une petite façade pour :
- estimer un coût "bits" (R) à partir d'un payload (proxy simple),
- calculer un score RD = D + lambda_rd * R en s'appuyant sur les métriques.

Ce module est volontairement léger pour Step 4; il sera étoffé ensuite:
- vrais estimateurs de bits symboliques (par tables rANS),
- pondération couture, multi-métriques, etc.
"""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import torch

try:
    from pc15metrics.psnr_ssim import ssim as _ssim
except Exception:
    _ssim = None  # fallback numpy


MetricName = Literal["ssim", "mse", "mixed"]


@dataclass(frozen=True)
class RDConfig:
    """Configuration minimaliste pour un score RD."""
    lambda_rd: float = 0.02
    metric: MetricName = "ssim"
    alpha: float = 0.7  # pour "mixed"


def bits_proxy_from_payload(payload: bytes) -> float:
    """
    Estimation **proxy** des bits d'un payload : `8 * len(payload)`.
    Suffisant pour Step 4; remplacé plus tard par un estimateur symbolique.
    """
    return float(len(payload) * 8)


def distortion(
    y: torch.Tensor,
    yhat: torch.Tensor,
    metric: MetricName = "ssim",
    alpha: float = 0.7,
) -> torch.Tensor:
    """
    Calcule D sur batch (B,1,H,W) → (B,).

    - "ssim"  → 1 - SSIM(y, ŷ)  (torch si dispo, sinon MSE proxy)
    - "mse"   → mean((y - ŷ)^2)
    - "mixed" → alpha*(1-SSIM) + (1-alpha)*MSE
    """
    assert y.shape == yhat.shape and y.ndim == 4 and y.shape[1] == 1
    B = y.shape[0]

    if metric == "mse" or _ssim is None:
        D = torch.mean((y - yhat) ** 2, dim=(1, 2, 3))
        if metric == "mixed":
            D = alpha * D + (1.0 - alpha) * D  # même terme (fallback)
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
