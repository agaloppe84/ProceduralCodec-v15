from __future__ import annotations
from dataclasses import dataclass
import torch
from pc15metrics.psnr_ssim import ssim

@dataclass(frozen=True)
class SearchCfg:
    lambda_rd: float = 0.02
    metric: str = "ssim"
    alpha: float = 0.7
    beam_k: int = 0
    early_exit_tau: float | None = 0.02

@dataclass
class ScoreOut:
    D: torch.Tensor
    R: torch.Tensor
    RD: torch.Tensor

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
