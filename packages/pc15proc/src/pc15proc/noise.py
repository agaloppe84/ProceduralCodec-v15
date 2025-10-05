from __future__ import annotations

import math
import torch
from pc15core.rng import to_int64_signed

# ---------------------------------------------------------------------
# Constantes & helpers bitwise (tout en int64 signé pour éviter l'overflow)
# ---------------------------------------------------------------------

# Mix constants (splitmix64-like), remappées en int64 signé (même bits)
_C1 = to_int64_signed(0xBF58476D1CE4E5B9)
_C2 = to_int64_signed(0x94D049BB133111EB)
# "golden ratio" 64-bit pour dé-corréler les flux
_GOLDEN64_I = to_int64_signed(0x9E3779B97F4A7C15)
# seed de base
_BASE_SEED_I = to_int64_signed(0x1234ABCD9876EF01)
# masque 53 bits pour une mantisse double
_M53 = (1 << 53) - 1


def _mix64(x: torch.Tensor) -> torch.Tensor:
    """SplitMix64 like mix, sur tenseur int64 (vectorisé)."""
    x = x ^ (x >> 30)
    x = x * _C1
    x = x ^ (x >> 27)
    x = x * _C2
    x = x ^ (x >> 31)
    return x


@torch.no_grad()
def _hash2(ix: torch.Tensor, iy: torch.Tensor, seed64: torch.Tensor) -> torch.Tensor:
    """Hash 2D (ix,iy) + seed -> int64 pseudo-aléatoire stable (vectorisé)."""
    # Toutes les entrées doivent être int64
    ix = ix.to(torch.int64)
    iy = iy.to(torch.int64)
    s = seed64.to(torch.int64)

    # Combine & mix
    h = _mix64(ix ^ _GOLDEN64_I)
    h = _mix64(h ^ (iy + (s ^ _BASE_SEED_I)))
    return h  # int64


@torch.no_grad()
def rand01(h: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Map int64 hash -> float aléatoire uniforme dans [0,1).
    Retourne float32 par défaut (ou `dtype` demandé).
    """
    if dtype is None:
        dtype = torch.float32
    x = h.to(torch.int64)
    # On prend 53 bits "positifs" (mantisse double) pour une bonne granularité,
    # puis on normalise en [0,1).
    mant = (x >> 11) & _M53
    out = mant.to(torch.float64) / float(1 << 53)
    return out.to(dtype=dtype)


# ---------------------------------------------------------------------
# Worley / Voronoi helpers (utilisés par plusieurs générateurs)
# ---------------------------------------------------------------------

@torch.no_grad()
def worley_f1(
    xx: torch.Tensor,
    yy: torch.Tensor,
    scale: torch.Tensor,
    seed64: torch.Tensor,
    *,
    metric: str = "euclidean",
) -> torch.Tensor:
    """Worley F1 distance (distance au site le plus proche).
    - xx, yy: grilles [-1,1] (typiquement shape [1,h,w])
    - scale: [B,1,1] ou scalaire broadcastable
    - seed64: [B,1,1] int64
    Retour: [B,h,w] (même dtype que xx/yy).
    """
    # dtype/shape
    dtype = xx.dtype
    device = xx.device
    B = scale.shape[0] if scale.ndim >= 3 else 1

    # Broadcast xx/yy vers [B,h,w]
    xxb = xx.expand(B, *xx.shape[-2:])
    yyb = yy.expand(B, *yy.shape[-2:])
    sc = scale.to(dtype=dtype)

    # Coordonnées de cellule
    fx = (xxb + 1.0) * (sc / 2.0)
    fy = (yyb + 1.0) * (sc / 2.0)
    xi = torch.floor(fx).to(torch.int64)
    yi = torch.floor(fy).to(torch.int64)
    xf = fx - xi.to(dtype)
    yf = fy - yi.to(dtype)

    best_d = torch.full_like(xxb, float("inf"))

    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            cx = xi + ox
            cy = yi + oy
            h = _hash2(cx, cy, seed64)
            # Jitter dans la cellule (uniforme [0,1))
            jx = rand01(h, dtype=dtype)
            jy = rand01(h ^ _GOLDEN64_I, dtype=dtype)  # ⚠️ pas de torch.tensor(...)

            dx = (cx.to(dtype) + jx) - fx
            dy = (cy.to(dtype) + jy) - fy

            if metric == "manhattan":
                d = dx.abs() + dy.abs()
            elif metric == "chebyshev":
                d = torch.maximum(dx.abs(), dy.abs())
            else:  # euclidean (par défaut)
                d = torch.sqrt(dx * dx + dy * dy)

            best_d = torch.minimum(best_d, d)

    return best_d  # [B,h,w], dtype de xx/yy


@torch.no_grad()
def worley_f2(
    xx: torch.Tensor,
    yy: torch.Tensor,
    scale: torch.Tensor,
    seed64: torch.Tensor,
    *,
    metric: str = "euclidean",
) -> torch.Tensor:
    """Worley F2 (deuxième distance la plus proche)."""
    dtype = xx.dtype
    B = scale.shape[0] if scale.ndim >= 3 else 1

    xxb = xx.expand(B, *xx.shape[-2:])
    yyb = yy.expand(B, *yy.shape[-2:])
    sc = scale.to(dtype=dtype)

    fx = (xxb + 1.0) * (sc / 2.0)
    fy = (yyb + 1.0) * (sc / 2.0)
    xi = torch.floor(fx).to(torch.int64)
    yi = torch.floor(fy).to(torch.int64)
    xf = fx - xi.to(dtype)
    yf = fy - yi.to(dtype)

    ds = []
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            cx = xi + ox
            cy = yi + oy
            h = _hash2(cx, cy, seed64)
            jx = rand01(h, dtype=dtype)
            jy = rand01(h ^ _GOLDEN64_I, dtype=dtype)

            dx = (cx.to(dtype) + jx) - fx
            dy = (cy.to(dtype) + jy) - fy

            if metric == "manhattan":
                d = dx.abs() + dy.abs()
            elif metric == "chebyshev":
                d = torch.maximum(dx.abs(), dy.abs())
            else:
                d = torch.sqrt(dx * dx + dy * dy)

            ds.append(d)

    D = torch.stack(ds, dim=0)  # [9,B,h,w]
    d_sorted, _ = torch.sort(D, dim=0)
    return d_sorted[1]  # F2


# ---------------------------------------------------------------------
# Value noise & Perlin 2D (utilisés par plusieurs générateurs)
# ---------------------------------------------------------------------

@torch.no_grad()
def value2d(xx: torch.Tensor, yy: torch.Tensor, scale: torch.Tensor, seed64: torch.Tensor) -> torch.Tensor:
    """Value noise 2D basique (bilerp sur valeurs pseudo-aléatoires aux coins)."""
    dtype = xx.dtype
    B = scale.shape[0] if scale.ndim >= 3 else 1

    xxb = xx.expand(B, *xx.shape[-2:])
    yyb = yy.expand(B, *yy.shape[-2:])
    sc = scale.to(dtype=dtype)

    fx = (xxb + 1.0) * (sc / 2.0)
    fy = (yyb + 1.0) * (sc / 2.0)
    xi = torch.floor(fx).to(torch.int64)
    yi = torch.floor(fy).to(torch.int64)
    xf = fx - xi.to(dtype)
    yf = fy - yi.to(dtype)

    def v(ix, iy):
        return rand01(_hash2(ix, iy, seed64), dtype=dtype) * 2.0 - 1.0

    v00 = v(xi, yi)
    v10 = v(xi + 1, yi)
    v01 = v(xi, yi + 1)
    v11 = v(xi + 1, yi + 1)

    # lissage (fade)
    def fade(t):  # 6t^5 - 15t^4 + 10t^3
        return t * t * t * (t * (t * 6 - 15) + 10)

    u = fade(xf.clamp(0, 1))
    v_ = fade(yf.clamp(0, 1))

    # bilerp
    a = v00 + u * (v10 - v00)
    b = v01 + u * (v11 - v01)
    out = a + v_ * (b - a)
    return out.clamp(-1, 1)


@torch.no_grad()
def perlin2d(xx: torch.Tensor, yy: torch.Tensor, scale: torch.Tensor, seed64: torch.Tensor) -> torch.Tensor:
    """Perlin 2D (gradient noise)."""
    dtype = xx.dtype
    B = scale.shape[0] if scale.ndim >= 3 else 1

    xxb = xx.expand(B, *xx.shape[-2:])
    yyb = yy.expand(B, *yy.shape[-2:])
    sc = scale.to(dtype=dtype)

    fx = (xxb + 1.0) * (sc / 2.0)
    fy = (yyb + 1.0) * (sc / 2.0)
    xi = torch.floor(fx).to(torch.int64)
    yi = torch.floor(fy).to(torch.int64)
    xf = fx - xi.to(dtype)
    yf = fy - yi.to(dtype)

    # gradients aux coins: angle ~ U[0, 2π)
    def grad(ix, iy):
        h = _hash2(ix, iy, seed64)
        ang = rand01(h, dtype=torch.float32) * (2.0 * math.pi)
        gx = torch.cos(ang).to(dtype=dtype)
        gy = torch.sin(ang).to(dtype=dtype)
        return gx, gy

    g00x, g00y = grad(xi, yi)
    g10x, g10y = grad(xi + 1, yi)
    g01x, g01y = grad(xi, yi + 1)
    g11x, g11y = grad(xi + 1, yi + 1)

    # vecteurs distance depuis les coins
    d00x, d00y = xf, yf
    d10x, d10y = xf - 1.0, yf
    d01x, d01y = xf, yf - 1.0
    d11x, d11y = xf - 1.0, yf - 1.0

    # produits scalaires
    n00 = g00x * d00x + g00y * d00y
    n10 = g10x * d10x + g10y * d10y
    n01 = g01x * d01x + g01y * d01y
    n11 = g11x * d11x + g11y * d11y

    # fade
    def fade(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    u = fade(xf.clamp(0, 1))
    v = fade(yf.clamp(0, 1))

    # interpolation bilinéaire
    a = n00 + u * (n10 - n00)
    b = n01 + u * (n11 - n01)
    out = a + v * (b - a)

    # normalisation douce (plage approx [-1,1])
    return out.clamp(-1, 1)


# ---------------------------------------------------------------------
# (Optionnel) octaves
# ---------------------------------------------------------------------

@torch.no_grad()
def fbm_perlin(
    xx: torch.Tensor,
    yy: torch.Tensor,
    scale: torch.Tensor,
    seed64: torch.Tensor,
    *,
    octaves: int = 4,
    gain: float = 0.5,
    lacunarity: float = 2.0,
) -> torch.Tensor:
    """fBM basé sur Perlin."""
    dtype = xx.dtype
    B = scale.shape[0] if scale.ndim >= 3 else 1
    amp = 1.0
    sc = scale.clone().to(dtype=dtype)
    s = seed64
    out = torch.zeros((B, *xx.shape[-2:]), device=xx.device, dtype=dtype)
    for _ in range(octaves):
        out = out + amp * perlin2d(xx, yy, sc, s)
        amp *= gain
        sc = sc * lacunarity
        s = s ^ _GOLDEN64_I
    # clamp doux
    return out.clamp(-1, 1)


@torch.no_grad()
def turbulence_value(
    xx: torch.Tensor,
    yy: torch.Tensor,
    scale: torch.Tensor,
    seed64: torch.Tensor,
    *,
    octaves: int = 4,
    gain: float = 0.5,
    lacunarity: float = 2.0,
) -> torch.Tensor:
    """Turbulence (somme des |value noise|)."""
    dtype = xx.dtype
    B = scale.shape[0] if scale.ndim >= 3 else 1
    amp = 1.0
    sc = scale.clone().to(dtype=dtype)
    s = seed64
    out = torch.zeros((B, *xx.shape[-2:]), device=xx.device, dtype=dtype)
    for _ in range(octaves):
        v = value2d(xx, yy, sc, s).abs()
        out = out + amp * v
        amp *= gain
        sc = sc * lacunarity
        s = s ^ _GOLDEN64_I
    # normalisation soft vers [-1,1]
    out = out / (out.amax(dim=(1, 2), keepdim=True).clamp(min=1e-6))
    return out * 2.0 - 1.0
