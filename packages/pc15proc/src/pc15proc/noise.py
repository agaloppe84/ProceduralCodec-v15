from __future__ import annotations
import math
import torch
from pc15core.rng import to_int64_signed

# Constante u64 remappée en int64 signé (mêmes bits) pour éviter l'overflow sur CPU
_GOLDEN64_I = to_int64_signed(0x9E3779B97F4A7C15)


@torch.no_grad()
def _hash2(x: torch.Tensor, y: torch.Tensor, seed: torch.Tensor) -> torch.Tensor:
    """Petit hash 2D déterministe (int64), compatible avec l'ancien code."""
    ix = x.to(torch.int64)
    iy = y.to(torch.int64)
    s = seed.to(torch.int64)

    # Constantes 32-bit (sûres en int64 signé)
    n = s ^ (ix * 0x27D4EB2D) ^ (iy * 0x9E3779B1)
    n = (n ^ (n >> 15)) * 0x85EBCA6B
    n = (n ^ (n >> 13)) * 0xC2B2AE35
    n = n ^ (n >> 16)
    return n  # int64


@torch.no_grad()
def rand01(seed64_xy: torch.Tensor) -> torch.Tensor:
    """Map hash int64 -> float32 uniforme [0,1)."""
    u32 = (seed64_xy & 0xFFFFFFFF).to(torch.int64)
    return (u32.to(torch.float32) / float(2**32)).to(torch.float32)


@torch.no_grad()
def _fade(t: torch.Tensor) -> torch.Tensor:
    # 6t^5 - 15t^4 + 10t^3
    return t * t * t * (t * (t * 6 - 15) + 10)


@torch.no_grad()
def _lerp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return a + (b - a) * t


# ---------------------------------------------------------------------
# Value noise (nom historique conservé)
# ---------------------------------------------------------------------
@torch.no_grad()
def value_noise2d(xx: torch.Tensor, yy: torch.Tensor, scale: torch.Tensor, seed64: torch.Tensor) -> torch.Tensor:
    """Value noise bilinéaire (retour ~ [0,1])."""
    fx = (xx + 1.0) * (scale / 2.0)
    fy = (yy + 1.0) * (scale / 2.0)
    xi = torch.floor(fx).to(torch.int64)
    yi = torch.floor(fy).to(torch.int64)
    xf = fx - xi.to(xx.dtype)
    yf = fy - yi.to(xx.dtype)

    def v(ix, iy):
        h = _hash2(ix, iy, seed64)
        return rand01(h)

    v00 = v(xi + 0, yi + 0)
    v10 = v(xi + 1, yi + 0)
    v01 = v(xi + 0, yi + 1)
    v11 = v(xi + 1, yi + 1)

    u = _fade(xf.clamp(0, 1))
    v_ = _fade(yf.clamp(0, 1))
    x1 = _lerp(v00, v10, u)
    x2 = _lerp(v01, v11, u)
    return _lerp(x1, x2, v_).clamp(0, 1)


# ---------------------------------------------------------------------
# Perlin 2D (noms historiques conservés)
# ---------------------------------------------------------------------
@torch.no_grad()
def perlin2d(xx: torch.Tensor, yy: torch.Tensor, scale: torch.Tensor, seed64: torch.Tensor) -> torch.Tensor:
    """Perlin 2D (retour ~ [-1,1])."""
    fx = (xx + 1.0) * (scale / 2.0)
    fy = (yy + 1.0) * (scale / 2.0)
    xi = torch.floor(fx).to(torch.int64)
    yi = torch.floor(fy).to(torch.int64)
    xf = fx - xi.to(xx.dtype)
    yf = fy - yi.to(xx.dtype)

    def grad(ix, iy):
        # angle ~ U[0, 2π)
        h = _hash2(ix, iy, seed64)
        ang = rand01(h).to(torch.float64) * (2.0 * math.pi)
        gx = torch.cos(ang).to(dtype=xx.dtype, device=xx.device)
        gy = torch.sin(ang).to(dtype=xx.dtype, device=xx.device)
        return gx, gy

    g00x, g00y = grad(xi + 0, yi + 0)
    g10x, g10y = grad(xi + 1, yi + 0)
    g01x, g01y = grad(xi + 0, yi + 1)
    g11x, g11y = grad(xi + 1, yi + 1)

    d00 = g00x * (xf - 0) + g00y * (yf - 0)
    d10 = g10x * (xf - 1) + g10y * (yf - 0)
    d01 = g01x * (xf - 0) + g01y * (yf - 1)
    d11 = g11x * (xf - 1) + g11y * (yf - 1)

    u = _fade(xf.clamp(0, 1))
    v = _fade(yf.clamp(0, 1))
    x1 = _lerp(d00, d10, u)
    x2 = _lerp(d01, d11, u)
    out = _lerp(x1, x2, v)
    return out.clamp(-1, 1)


# ---------------------------------------------------------------------
# Worley F1 (utilisé par plusieurs générateurs)
# ---------------------------------------------------------------------
@torch.no_grad()
def worley_f1(
    xx: torch.Tensor,
    yy: torch.Tensor,
    scale: torch.Tensor,
    seed64: torch.Tensor,
    metric: str = "euclidean",
) -> torch.Tensor:
    """Worley F1 (distance au site le plus proche)."""
    fx = (xx + 1.0) * (scale / 2.0)
    fy = (yy + 1.0) * (scale / 2.0)
    xi = torch.floor(fx).to(torch.int64)
    yi = torch.floor(fy).to(torch.int64)
    xf = fx - xi.to(xx.dtype)
    yf = fy - yi.to(xx.dtype)

    ds = []
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            cx = xi + ox
            cy = yi + oy
            h = _hash2(cx, cy, seed64)
            jx = rand01(h)
            # ⚠️ plus de torch.tensor(u64) → XOR avec int64 signé sûr
            jy = rand01(h ^ _GOLDEN64_I)

            px = (ox + jx)
            py = (oy + jy)
            dx = (xf - px)
            dy = (yf - py)

            if metric == "manhattan":
                d = torch.abs(dx) + torch.abs(dy)
            elif metric == "chebyshev":
                d = torch.maximum(torch.abs(dx), torch.abs(dy))
            else:
                d = torch.sqrt(dx * dx + dy * dy)

            ds.append(d)

    dmin, _ = torch.stack(ds, dim=0).min(dim=0)
    # même normalisation que ton ancien code
    return (dmin / 1.5).clamp(0, 1)
