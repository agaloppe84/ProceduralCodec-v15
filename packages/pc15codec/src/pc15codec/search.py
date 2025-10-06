from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence, Union
import numpy as np
import torch
from pc15metrics.psnr_ssim import ssim


@dataclass(frozen=True)
class SearchCfg:
    lambda_rd: float = 0.02
    metric: str = "ssim"   # "ssim" | "mse" (ou autre futur)
    alpha: float = 0.7
    beam_k: int = 0
    early_exit_tau: float | None = 0.02


@dataclass
class ScoreOut:
    D: torch.Tensor    # (B,)
    R: torch.Tensor    # (B,)
    RD: torch.Tensor   # (B,)


def make_border_mask(h: int, w: int, width: int = 8, device=None, dtype=None):
    y = torch.zeros((1, 1, h, w), device=device, dtype=dtype)
    y[..., :width, :] = 1
    y[..., -width:, :] = 1
    y[..., :, :width] = 1
    y[..., :, -width:] = 1
    return y


def score_batch(
    y_tile: torch.Tensor,
    synth: torch.Tensor,
    cfg: SearchCfg,
    seam_weight: float = 0.0,
    border_mask: torch.Tensor | None = None,
) -> ScoreOut:
    """
    Score RD basique sur un batch de synthèses.
    - D = 1-SSIM (ou MSE) [+ pénalité couture optionnelle]
    - R = 1.0 (constante)  → utile quand on ne dispose pas encore d'une estimation des bits
    - RD = D + λ·R
    """
    B = synth.shape[0]
    yB = y_tile.expand(B, -1, -1, -1)

    if cfg.metric == "ssim":
        D = 1.0 - ssim(yB, synth)
    else:
        D = torch.mean((yB - synth) ** 2, dim=(1, 2, 3))

    if seam_weight > 0.0:
        if border_mask is None:
            h, w = y_tile.shape[-2:]
            border_mask = make_border_mask(h, w, width=8, device=y_tile.device, dtype=y_tile.dtype)
        seam = torch.mean(((yB - synth) ** 2) * border_mask, dim=(1, 2, 3))
        D = D + seam_weight * seam

    R = torch.full_like(D, 1.0)
    RD = D + cfg.lambda_rd * R
    return ScoreOut(D=D, R=R, RD=RD)


# ---------------------------------------------------------------------------
# Extensions : bits estimés + helper CPU-friendly (numpy)
# ---------------------------------------------------------------------------

def _as_batched_bits(bits: Union[float, Sequence[float], torch.Tensor],
                     B: int, device=None, dtype=None) -> torch.Tensor:
    """Normalise une estimation de bits (scalaire/liste/tensor) en tensor (B,)."""
    if isinstance(bits, torch.Tensor):
        if bits.numel() == 1:
            return torch.full((B,), float(bits.item()), device=device, dtype=dtype)
        t = bits.to(device=device, dtype=dtype).reshape(-1)
        assert t.numel() == B, "bits_est length must match batch size"
        return t
    if isinstance(bits, (list, tuple)):
        assert len(bits) == B, "bits_est length must match batch size"
        return torch.tensor(bits, device=device, dtype=dtype)
    return torch.full((B,), float(bits), device=device, dtype=dtype)


def score_batch_bits(
    y_tile: torch.Tensor,
    synth: torch.Tensor,
    cfg: SearchCfg,
    bits_est: Optional[Union[float, Sequence[float], torch.Tensor]] = None,
    seam_weight: float = 0.0,
    border_mask: torch.Tensor | None = None,
) -> ScoreOut:
    """
    Variante de score_batch qui prend une estimation de bits (R) par candidat.
    - bits_est : float | [float]*B | torch.Tensor(B)
    """
    B = synth.shape[0]
    yB = y_tile.expand(B, -1, -1, -1)

    if cfg.metric == "ssim":
        D = 1.0 - ssim(yB, synth)
    else:
        D = torch.mean((yB - synth) ** 2, dim=(1, 2, 3))

    if seam_weight > 0.0:
        if border_mask is None:
            h, w = y_tile.shape[-2:]
            border_mask = make_border_mask(h, w, width=8, device=y_tile.device, dtype=y_tile.dtype)
        seam = torch.mean(((yB - synth) ** 2) * border_mask, dim=(1, 2, 3))
        D = D + seam_weight * seam

    if bits_est is None:
        R = torch.full_like(D, 1.0)
    else:
        R = _as_batched_bits(bits_est, B, device=D.device, dtype=D.dtype)

    RD = D + cfg.lambda_rd * R
    return ScoreOut(D=D, R=R, RD=RD)


def score_rd_numpy(
    y: np.ndarray,
    yhat: np.ndarray,
    lam: float,
    alpha: float,
    bits_est: float,
    metric: str = "ssim",
) -> float:
    """
    Score RD CPU-friendly pour prototypage/tests.
    - Si metric='ssim': on tente d'utiliser ssim torch (1x1HxW). Si indispo, fallback ~ 0.5*MSE.
    - Sinon: MSE pur.
    """
    y = np.asarray(y, dtype=np.float32)
    yhat = np.asarray(yhat, dtype=np.float32)

    if metric == "ssim":
        try:
            Y = torch.from_numpy(y).reshape(1, 1, *y.shape[-2:])
            YH = torch.from_numpy(yhat).reshape(1, 1, *yhat.shape[-2:])
            D_t = 1.0 - ssim(Y, YH)
            D = float(D_t.reshape(-1)[0])
        except Exception:
            mse = float(np.mean((y - yhat) ** 2))
            D = 0.5 * mse  # proxy raisonnable
    else:
        D = float(np.mean((y - yhat) ** 2))

    R = float(bits_est)
    return float(D + lam * R)
