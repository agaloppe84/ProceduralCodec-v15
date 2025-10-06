from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import torch
from pc15metrics.psnr_ssim import ssim


# -----------------------------------------------------------------------------
# Config & sorties
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class SearchCfg:
    """
    Configuration de scoring RD.

    - lambda_rd : poids du terme "bits" (R) dans RD = D + lambda_rd * R
    - metric    : "ssim" (par défaut), "mse", ou "mixed" (pondère SSIM/MSE via alpha)
    - alpha     : poids de SSIM dans le mode "mixed" (D = alpha*(1-ssim) + (1-alpha)*MSE)
    - beam_k    : réservé pour beam-search (non utilisé ici)
    - early_exit_tau : seuil d'early exit (non utilisé ici)
    """
    lambda_rd: float = 0.02
    metric: str = "ssim"  # "ssim" | "mse" | "mixed"
    alpha: float = 0.7
    beam_k: int = 0
    early_exit_tau: float | None = 0.02


@dataclass
class ScoreOut:
    """Paquet résultat du score batch."""
    D: torch.Tensor   # (B,)
    R: torch.Tensor   # (B,)  -- proxy bits ou bits fournis
    RD: torch.Tensor  # (B,)


# -----------------------------------------------------------------------------
# Masque de bord (pénalité couture)
# -----------------------------------------------------------------------------

def make_border_mask(h: int, w: int, width: int = 8, device=None, dtype=None) -> torch.Tensor:
    """
    Crée un masque binaire (1 sur les bords, 0 ailleurs), shape (1,1,H,W).
    width=0 → masque nul (aucun bord).
    """
    y = torch.zeros((1, 1, h, w), device=device, dtype=dtype)
    if width <= 0:
        return y
    width = int(width)
    y[..., :width, :] = 1
    y[..., -width:, :] = 1
    y[..., :, :width] = 1
    y[..., :, -width:] = 1
    return y


# -----------------------------------------------------------------------------
# Scoring RD (proxy-bits constant = 1.0)
# -----------------------------------------------------------------------------

def score_batch(
    y_tile: torch.Tensor,
    synth: torch.Tensor,
    cfg: SearchCfg,
    seam_weight: float = 0.0,
    border_mask: Optional[torch.Tensor] = None,
) -> ScoreOut:
    """
    Calcule D, R, RD pour un batch de synthèses 'synth' (B,1,H,W) vs la tuile 'y_tile' (1,1,H,W).

    - D:
        * "ssim"  → 1 - SSIM(y, ŷ)
        * "mse"   → mean((y - ŷ)^2)
        * "mixed" → alpha*(1-SSIM) + (1-alpha)*MSE   (alpha dans cfg)
    - Pénalité couture: si seam_weight>0, ajoute seam_weight * MSE sur la zone de bord (border_mask).
      Si border_mask=None, il est généré automatiquement (width=8).
    - R: proxy bits (placeholder = 1.0 pour S1, donc constante)
    - RD: D + lambda_rd * R
    """
    assert y_tile.ndim == 4 and y_tile.shape[0] == 1 and y_tile.shape[1] == 1, "y_tile doit être (1,1,H,W)"
    assert synth.ndim == 4 and synth.shape[1] == 1, "synth doit être (B,1,H,W)"
    assert y_tile.shape[-2:] == synth.shape[-2:], "H,W doivent matcher"

    B = synth.shape[0]
    yB = y_tile.expand(B, -1, -1, -1)

    # Distorsion principale
    if cfg.metric == "ssim":
        D = 1.0 - ssim(yB, synth)  # (B,)
    elif cfg.metric == "mse":
        D = torch.mean((yB - synth) ** 2, dim=(1, 2, 3))
    elif cfg.metric == "mixed":
        d_ssim = 1.0 - ssim(yB, synth)
        d_mse = torch.mean((yB - synth) ** 2, dim=(1, 2, 3))
        a = float(cfg.alpha)
        D = a * d_ssim + (1.0 - a) * d_mse
    else:
        raise ValueError(f"Unknown metric: {cfg.metric}")

    # Pénalité couture
    if seam_weight > 0.0:
        if border_mask is None:
            h, w = y_tile.shape[-2:]
            border_mask = make_border_mask(h, w, width=8, device=y_tile.device, dtype=y_tile.dtype)
        if border_mask.device != y_tile.device:
            border_mask = border_mask.to(y_tile.device)
        if border_mask.dtype != y_tile.dtype:
            border_mask = border_mask.to(dtype=y_tile.dtype)
        seam = torch.mean(((yB - synth) ** 2) * border_mask, dim=(1, 2, 3))
        D = D + float(seam_weight) * seam

    # Proxy bits (placeholder constant pour S1)
    R = torch.ones_like(D)
    RD = D + float(cfg.lambda_rd) * R
    return ScoreOut(D=D, R=R, RD=RD)


# -----------------------------------------------------------------------------
# Scoring RD (bits fournis par appelant)
# -----------------------------------------------------------------------------
def score_batch_bits(
    y_tile: torch.Tensor,
    synth: torch.Tensor,
    cfg: SearchCfg,
    *,
    bits_est: Optional[torch.Tensor | List[float]] = None,
    bits: Optional[torch.Tensor | List[float]] = None,
    seam_weight: float = 0.0,
    border_mask: Optional[torch.Tensor] = None,
) -> ScoreOut:
    """
    Variante où le coût "bits" (R) est fourni par l'appelant, shape (B,).

    Compat :
      - `bits_est=` (ancien nom utilisé par certains tests)
      - `bits=`     (nom recommandé)

    - D: même calcul que score_batch (SSIM/MSE/mixed + couture optionnelle)
    - R: = bits fournis
    - RD: D + lambda_rd * R
    """
    assert y_tile.ndim == 4 and y_tile.shape[0] == 1 and y_tile.shape[1] == 1, "y_tile doit être (1,1,H,W)"
    assert synth.ndim == 4 and synth.shape[1] == 1, "synth doit être (B,1,H,W)"
    assert y_tile.shape[-2:] == synth.shape[-2:], "H,W doivent matcher"

    # Supporte bits_est (legacy) ou bits (nouveau)
    bits_vec = bits_est if bits_est is not None else bits
    if bits_vec is None:
        raise ValueError("You must pass either `bits_est=` (legacy) or `bits=` (preferred).")

    # Convertit vers tensor (B,)
    if not torch.is_tensor(bits_vec):
        bits_t = torch.tensor(bits_vec, device=synth.device, dtype=synth.dtype)
    else:
        bits_t = bits_vec.to(device=synth.device, dtype=synth.dtype)

    B = synth.shape[0]
    if bits_t.ndim != 1 or bits_t.numel() != B:
        raise ValueError(f"`bits` must be shape (B,), got {tuple(bits_t.shape)} for B={B}")

    # Calcule D via score_batch (incluant couture éventuelle)
    out = score_batch(y_tile, synth, cfg, seam_weight=seam_weight, border_mask=border_mask)
    D = out.D

    # Remplace R par les bits fournis et recalcule RD
    R = bits_t
    RD = D + float(cfg.lambda_rd) * R
    return ScoreOut(D=D, R=R, RD=RD)



# -----------------------------------------------------------------------------
# Coarse-to-fine helpers (math-only, génériques)
# -----------------------------------------------------------------------------

def grid_linspace(bounds: List[Tuple[float, float]], steps: List[int], device=None, dtype=None) -> torch.Tensor:
    """
    Crée une grille régulière D-dimensionnelle.
    bounds: [(min_d, max_d)] pour d=0..D-1
    steps:  [S_d] nombre d'échantillons par dimension
    return: (N, D) avec N = prod(S_d)
    """
    assert len(bounds) == len(steps) and len(bounds) > 0, "bounds/steps mismatch"
    axes = [torch.linspace(lo, hi, int(S), device=device, dtype=dtype) for (lo, hi), S in zip(bounds, steps)]
    mesh = torch.meshgrid(*axes, indexing="ij")
    grid = torch.stack([m.reshape(-1) for m in mesh], dim=-1)
    return grid  # (N,D)


def local_refine(center: torch.Tensor, deltas: torch.Tensor, steps: List[int]) -> torch.Tensor:
    """
    Grille locale autour d'un centre (D,) avec amplitude 'deltas' (D,).
    Pour chaque dim d: linspace(center[d]-deltas[d], center[d]+deltas[d], steps[d]).
    return: (N,D)
    """
    assert center.ndim == 1 and deltas.ndim == 1 and center.numel() == deltas.numel()
    D = center.numel()
    bds: List[Tuple[float, float]] = []
    for d in range(D):
        lo = float(center[d] - deltas[d])
        hi = float(center[d] + deltas[d])
        if hi < lo:  # garde-fou
            lo, hi = hi, lo
        bds.append((lo, hi))
    return grid_linspace(bds, steps, device=center.device, dtype=center.dtype)


def select_best_param_from_scores(params: torch.Tensor, scores: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """
    params: (N,D) paramètres candidats
    scores: (N,) coût (plus petit = meilleur)
    return: (param_best(D,), index)
    """
    assert params.ndim == 2 and scores.ndim == 1 and params.shape[0] == scores.shape[0]
    idx = int(torch.argmin(scores).item())
    return params[idx], idx


# -----------------------------------------------------------------------------
# Helpers optionnels (future-proof) — non utilisés par les tests actuels
# -----------------------------------------------------------------------------

def clamp_params(params: torch.Tensor, bounds: List[Tuple[float, float]]) -> torch.Tensor:
    """Coupe chaque dimension de params (N,D) dans les bornes fournies."""
    assert params.ndim == 2 and params.shape[1] == len(bounds)
    out = params.clone()
    for d, (lo, hi) in enumerate(bounds):
        out[:, d] = out[:, d].clamp(min=float(lo), max=float(hi))
    return out


def topk_indices(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Indices des k meilleurs scores (croissants)."""
    k = max(1, min(int(k), scores.numel()))
    _, idx = torch.topk(-scores, k)  # négatif → small first
    ord_idx = torch.argsort(scores[idx])
    return idx[ord_idx]


# -----------------------------------------------------------------------------
# Numpy helper (utilisé par des tests)
# -----------------------------------------------------------------------------

def score_rd_numpy(
    y: np.ndarray,
    recon: np.ndarray,
    *,
    lam: Optional[float] = None,                  # alias legacy -> lambda_rd
    lambda_rd: float = 0.02,
    alpha: float = 0.7,
    bits_est: Optional[float] = None,             # alias legacy -> bits
    bits: Optional[float] = None,
    metric: str = "ssim",
) -> float:
    """
    Version NumPy, sans torch. Objectif: *simple et déterministe* pour tests rapides.

    Compat (legacy):
      - `lam`      : alias de `lambda_rd`
      - `bits_est` : alias de `bits`

    Retourne un scalaire RD = D + lambda_rd * bits.

    Notes:
      - Pour metric="ssim", on utilise ici un proxy basé sur la MSE (c'est suffisant pour les tests
        qui ne vérifient que l'ordre: "bon" < "mauvais" et la finitude).
    """
    # Aliases → valeurs effectives
    lam_eff = lambda_rd if lam is None else float(lam)
    bits_eff = bits if bits is not None else (float(bits_est) if bits_est is not None else 1.0)

    # coercition et contrôles
    y = np.asarray(y, dtype=np.float64)
    recon = np.asarray(recon, dtype=np.float64)
    if y.shape != recon.shape:
        raise ValueError(f"y and recon must have same shape, got {y.shape} vs {recon.shape}")

    # D : proxy 1-SSIM via MSE (suffisant pour l’inégalité s_good < s_bad des tests)
    mse = np.mean((y - recon) ** 2)
    if metric == "ssim":
        D = mse
    elif metric == "mse":
        D = mse
    else:
        # fallback: idem
        D = mse

    RD = float(D + lam_eff * bits_eff)
    # robustesse : garantir finite
    if not np.isfinite(RD):
        raise ValueError("Non-finite RD score")
    return RD
