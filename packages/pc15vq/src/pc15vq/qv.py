
from __future__ import annotations
from typing import Tuple
import torch

"""
pc15vq.qv — Quantification vectorielle (QV) minimaliste S1
---------------------------------------------------------
Objectif: fournir un codebook KxD et des primitives de quantif/déquantif robustes,
CPU-friendly, déterministes, pour intégrer QV dans le pipeline (S2+).

API:
- train_qv_kmeans(data, K, iters=20, seed=0) -> codebook (K,D)
- quantize_params(p, codebook) -> (idx, residual)          # p ≈ codebook[idx] + residual
- dequantize_params(idx, residual, codebook) -> p_recon
- quantize_batch(P, codebook) -> (indices, residuals)      # P:(N,D)
- dequantize_batch(indices, residuals, codebook) -> P_recon
"""

def _kmeans_init_pp(data: torch.Tensor, K: int, seed: int = 0) -> torch.Tensor:
    # data: (N,D)
    gen = torch.Generator(device=data.device)
    gen.manual_seed(int(seed))
    N, D = data.shape
    # Choix d'un premier centre au hasard
    idx0 = torch.randint(0, N, (1,), generator=gen, device=data.device)
    centers = data[idx0].clone()  # (1,D)

    # Init++
    for _ in range(1, K):
        # distances au centre le plus proche
        d2 = torch.cdist(data, centers, p=2)**2  # (N,C)
        mind2, _ = d2.min(dim=1)
        probs = mind2 / (mind2.sum() + 1e-12)
        # échantillonnage discret
        r = torch.rand((), generator=gen, device=data.device)
        cdf = probs.cumsum(dim=0)
        j = int(torch.searchsorted(cdf, r).item())
        centers = torch.cat([centers, data[j:j+1]], dim=0)
    return centers

def train_qv_kmeans(data: torch.Tensor, K: int, iters: int = 20, seed: int = 0) -> torch.Tensor:
    """
    data: (N,D) float32
    return: codebook: (K,D)
    """
    assert data.ndim == 2 and data.numel() > 0
    N, D = data.shape
    K = int(K)
    assert 1 <= K <= max(1, N), "invalid K"
    codebook = _kmeans_init_pp(data, K, seed=seed)

    for _ in range(int(iters)):
        # Assignation
        d2 = torch.cdist(data, codebook, p=2)**2  # (N,K)
        idx = d2.argmin(dim=1)                    # (N,)
        # Mise à jour
        new_cb = torch.zeros_like(codebook)
        counts = torch.zeros((K,), device=data.device, dtype=torch.long)
        new_cb.index_add_(0, idx, data)
        counts.index_add_(0, idx, torch.ones_like(idx, dtype=torch.long))
        # Éviter les clusters vides: garder l'ancien centre
        mask = counts > 0
        if mask.any():
            new_cb[mask] = new_cb[mask] / counts[mask].unsqueeze(1).to(new_cb.dtype)
        codebook = torch.where(mask.unsqueeze(1), new_cb, codebook)
    return codebook

def quantize_params(p: torch.Tensor, codebook: torch.Tensor) -> Tuple[int, torch.Tensor]:
    """
    p: (D,), codebook: (K,D)
    return: (idx, residual) tel que p ≈ codebook[idx] + residual
    """
    assert p.ndim == 1 and codebook.ndim == 2
    d2 = torch.cdist(p.unsqueeze(0), codebook, p=2)**2  # (1,K)
    idx = int(d2.argmin(dim=1).item())
    residual = p - codebook[idx]
    return idx, residual

def dequantize_params(idx: int, residual: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    return codebook[int(idx)] + residual

def quantize_batch(P: torch.Tensor, codebook: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    P: (N,D)
    return: (indices(N,), residuals(N,D))
    """
    assert P.ndim == 2 and codebook.ndim == 2
    d2 = torch.cdist(P, codebook, p=2)**2  # (N,K)
    idx = d2.argmin(dim=1)                 # (N,)
    residuals = P - codebook[idx]
    return idx, residuals

def dequantize_batch(indices: torch.Tensor, residuals: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    return codebook[indices] + residuals
