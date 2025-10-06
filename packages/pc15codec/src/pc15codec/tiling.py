from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import torch

@dataclass(frozen=True)
class TileGridCfg:
    """Configuration de grille pour le tiling/blend."""
    size: int = 256
    overlap: int = 24

    def __post_init__(self):
        if self.size <= 0:
            raise ValueError("size must be > 0")
        if self.overlap < 0 or self.overlap >= self.size:
            raise ValueError("overlap must be in [0, size-1]")

@dataclass(frozen=True)
class TileBatchSpec:
    """Spécification de batch de tuiles calculée pour HxW."""
    H: int
    W: int
    size: int
    overlap: int
    ny: int
    nx: int
    starts: List[Tuple[int, int]]  # [(y0, x0), ...] longueur = count

    @property
    def count(self) -> int:
        return len(self.starts)

def _start_indices(L: int, size: int, overlap: int) -> List[int]:
    """Positions de départ couvrant [0, L) avec overlap.
    Stride = size - overlap. Force la dernière tuile à L-size.
    """
    stride = max(1, size - overlap)
    starts = [0]
    s = 0
    while s + size < L:
        s += stride
        starts.append(s)
    last = max(0, L - size)
    if starts[-1] != last:
        starts.append(last)
    return starts

def tile_image(y: torch.Tensor, grid: TileGridCfg) -> TileBatchSpec:
    """Prépare la grille de tuiles pour une image Y [1,1,H,W]."""
    if y.ndim != 4 or y.shape[0] != 1 or y.shape[1] != 1:
        raise ValueError("y must be [1,1,H,W]")
    H, W = int(y.shape[2]), int(y.shape[3])
    ys = _start_indices(H, grid.size, grid.overlap)
    xs = _start_indices(W, grid.size, grid.overlap)
    starts = [(yy, xx) for yy in ys for xx in xs]
    return TileBatchSpec(H=H, W=W, size=grid.size, overlap=grid.overlap,
                         ny=len(ys), nx=len(xs), starts=starts)

# -----------------------------
# Fenêtres 1D : Hann / Tukey / Kaiser
# -----------------------------

def _hann_1d(n: int, overlap: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if overlap <= 0:
        return torch.ones(n, device=device, dtype=dtype)
    overlap = min(overlap, n // 2)
    w = torch.ones(n, device=device, dtype=dtype)
    t = torch.linspace(0, torch.pi, steps=overlap, device=device, dtype=dtype)
    ramp = 0.5 * (1 - torch.cos(t))  # 0..1
    w[:overlap] = ramp
    w[-overlap:] = torch.flip(ramp, dims=[0])
    return w

def _tukey_1d(n: int, alpha: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Fenêtre de Tukey (alpha in [0,1]) : plateau central + cos aux bords."""
    if alpha <= 0:
        return torch.ones(n, device=device, dtype=dtype)
    if alpha >= 1:
        # alpha=1 == Hann
        t = torch.linspace(0, torch.pi, steps=n, device=device, dtype=dtype)
        return 0.5 * (1 - torch.cos(t))
    w = torch.empty(n, device=device, dtype=dtype)
    L = n - 1
    x = torch.linspace(0, 1, steps=n, device=device, dtype=dtype)
    # zones : [0, a/2] montée cos, [a/2, 1-a/2] plateau 1, [1-a/2, 1] descente cos
    a = torch.tensor(alpha, device=device, dtype=dtype)
    left = x < (a / 2)
    right = x > (1 - a / 2)
    middle = (~left) & (~right)
    w[left] = 0.5 * (1 + torch.cos(torch.pi * (2 * x[left] / a - 1)))
    w[middle] = 1.0
    w[right] = 0.5 * (1 + torch.cos(torch.pi * (2 * x[right] / a - 2 / a + 1)))
    return w

def _kaiser_1d(n: int, beta: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Fenêtre de Kaiser. Beta ~ 6–8 donne des bords propres."""
    try:
        # dispo dans torch moderne
        return torch.kaiser_window(n, periodic=False, beta=beta, device=device, dtype=dtype)
    except Exception:
        # fallback simple avec i0 approx via torch.special
        x = torch.linspace(-1, 1, steps=n, device=device, dtype=dtype)
        from torch.special import i0
        denom = i0(torch.tensor(beta, device=device, dtype=dtype))
        return i0(beta * torch.sqrt(1 - x * x)) / denom

def _window_2d(size: int, overlap: int, device: torch.device, dtype: torch.dtype,
               kind: str = "hann", params: Optional[Dict[str, float]] = None) -> torch.Tensor:
    """Construit une fenêtre 2D séparée (wy ⊗ wx)."""
    if kind == "hann":
        wy = _hann_1d(size, overlap, device, dtype)
        wx = _hann_1d(size, overlap, device, dtype)
    elif kind == "tukey":
        # alpha proportionnel au ratio d'overlap
        alpha = (params or {}).get("alpha", max(0.0, min(1.0, 2.0 * overlap / max(1, size))))
        wy = _tukey_1d(size, alpha, device, dtype)
        wx = _tukey_1d(size, alpha, device, dtype)
    elif kind == "kaiser":
        beta = float((params or {}).get("beta", 6.5))
        wy = _kaiser_1d(size, beta, device, dtype)
        wx = _kaiser_1d(size, beta, device, dtype)
    else:
        raise ValueError(f"unknown window kind: {kind}")
    return (wy[:, None] * wx[None, :])  # [size,size]

# -----------------------------
# Seam penalty mask (pour RD)
# -----------------------------

def seam_penalty_mask(size: int, overlap: int, power: float = 1.0,
                      device: Optional[torch.device] = None,
                      dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Masque [size,size] proche de 1 aux bords, 0 au centre.
    À utiliser comme poids additionnel dans le score RD pour pénaliser les coutures.

    power>1 accentue la pénalité aux bords. Si overlap=0 → zeros.
    """
    device = device or torch.device("cpu")
    dtype = dtype or torch.float32
    if overlap <= 0:
        return torch.zeros((size, size), device=device, dtype=dtype)
    y = torch.arange(size, device=device, dtype=dtype)
    x = torch.arange(size, device=device, dtype=dtype)
    Y, X = torch.meshgrid(y, x, indexing="ij")
    d_edge = torch.minimum(torch.minimum(Y, X), torch.minimum(size - 1 - Y, size - 1 - X))
    m = torch.clamp(1.0 - d_edge / float(overlap), 0.0, 1.0)
    return m.pow(power)

# -----------------------------
# Blend (Partition of Unity)
# -----------------------------

@torch.no_grad()
def blend(tiles: torch.Tensor, spec: TileBatchSpec, H: int, W: int,
          window: str = "hann", window_params: Optional[Dict[str, float]] = None,
          eps: float = 1e-8) -> torch.Tensor:
    """Assemble des tuiles [N,1,size,size] en une image [1,1,H,W] avec blend normalisé.

    - window: 'hann' (defaut) | 'tukey' | 'kaiser'
    - window_params:
        - 'tukey': {'alpha': float in [0,1]} par défaut alpha=2*overlap/size clampé
        - 'kaiser': {'beta': float} par défaut 6.5
    """
    if tiles.ndim != 4 or tiles.shape[1] != 1:
        raise ValueError("tiles must be [N,1,h,w]")
    if tiles.shape[2] != spec.size or tiles.shape[3] != spec.size:
        raise ValueError("tile size mismatch with spec")
    if H != spec.H or W != spec.W:
        raise ValueError("H/W mismatch with spec")

    device, dtype = tiles.device, tiles.dtype
    out = torch.zeros((1, 1, H, W), device=device, dtype=dtype)
    acc = torch.zeros((1, 1, H, W), device=device, dtype=dtype)

    w2d = _window_2d(spec.size, spec.overlap, device, dtype, kind=window, params=window_params)  # [size,size]
    w2d = w2d.unsqueeze(0).unsqueeze(0)  # [1,1,size,size]

    N = tiles.shape[0]
    if N != spec.count:
        raise ValueError(f"tiles count ({N}) != spec.count ({spec.count})")

    # Accumulation pondérée + carte de couverture
    for i, (y0, x0) in enumerate(spec.starts):
        y1, x1 = y0 + spec.size, x0 + spec.size
        out[:, :, y0:y1, x0:x1] += tiles[i:i+1, :, :, :] * w2d
        acc[:, :, y0:y1, x0:x1] += w2d

    out = out / (acc + eps)  # Partition of Unity
    return out
