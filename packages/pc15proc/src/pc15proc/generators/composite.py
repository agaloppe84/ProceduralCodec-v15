from __future__ import annotations
import math
import torch

from ..api import Generator, GeneratorInfo, ParamSpec
from ..utils import grid
from ..noise import perlin2d, worley_f1
from ..registry import get as get_gen
from pc15core.rng import to_int64_signed

# Alphabets discrets (IDs stables -> bitstream stable)
_BASES = ("STRIPES", "CHECKER", "PERLIN", "WORLEY", "WOOD", "MARBLE")
_WARPS = ("NONE", "FBM", "RIPPLE", "SWIRL")
_MASKS = ("NONE", "VORONOI_EDGES", "DOTS", "HEX_GRID")
_PALS  = ("NONE", "PALETTE2", "PALETTE3")

# Constante int64 signée sûre
_GOLDEN = to_int64_signed(0x9E3779B97F4A7C15)


def _slot_seed(seed64: torch.Tensor, slot_id: int) -> torch.Tensor:
    """Dérive un seed par slot de manière déterministe (int64 signé)."""
    offs = to_int64_signed((0x85EBCA6B * (slot_id + 1)) & 0xFFFFFFFF)
    return (seed64.to(torch.int64) ^ _GOLDEN ^ offs)


def _smoothstep(x: torch.Tensor, edge0: float, edge1: float) -> torch.Tensor:
    t = ((x - edge0) / max(edge1 - edge0, 1e-6)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class Composite(Generator):
    @property
    def info(self) -> GeneratorInfo:
        # 4 slots discrets + 4 proxies continus (0..1)
        return GeneratorInfo(
            name="COMPOSITE",
            param_specs=(
                ParamSpec("base_id", "int", (0, len(_BASES) - 1)),
                ParamSpec("warp_id", "int", (0, len(_WARPS) - 1)),
                ParamSpec("mask_id", "int", (0, len(_MASKS) - 1)),
                ParamSpec("pal_id",  "int", (0, len(_PALS)  - 1)),
                ParamSpec("base_q",  "float", (0.0, 1.0)),
                ParamSpec("warp_q",  "float", (0.0, 1.0)),
                ParamSpec("mask_q",  "float", (0.0, 1.0)),
                ParamSpec("pal_q",   "float", (0.0, 1.0)),
            ),
            supports_noise=True,
        )

    @torch.no_grad()
    def render(self, tiles_hw, params, seeds, *, device, dtype):
        B = params.shape[0]
        h, w = tiles_hw
        xx, yy = grid(h, w, device=device, dtype=dtype)  # [1,h,w]
        xx = xx.expand(B, h, w).contiguous()
        yy = yy.expand(B, h, w).contiguous()

        # Decode params (shape-safe)
        base_id = params[:, 0].round().clamp(0, len(_BASES) - 1).to(torch.int64)
        warp_id = params[:, 1].round().clamp(0, len(_WARPS) - 1).to(torch.int64)
        mask_id = params[:, 2].round().clamp(0, len(_MASKS) - 1).to(torch.int64)
        pal_id  = params[:, 3].round().clamp(0, len(_PALS)  - 1).to(torch.int64)

        base_q = params[:, 4].to(dtype).view(B, 1, 1)
        warp_q = params[:, 5].to(dtype).view(B, 1, 1)
        mask_q = params[:, 6].to(dtype).view(B, 1, 1)
        pal_q  = params[:, 7].to(dtype).view(B, 1, 1)

        seed_tile = seeds.view(B, 1, 1).to(torch.int64)
        seed_base = _slot_seed(seed_tile, 1)
        seed_warp = _slot_seed(seed_tile, 2)
        seed_mask = _slot_seed(seed_tile, 3)
        seed_pal  = _slot_seed(seed_tile, 4)

        # ------------------------------------------------------------------
        # 1) BASE : on réutilise un générateur existant de la registry
        # ------------------------------------------------------------------
        outs = []
        for i in range(B):
            g = get_gen(_BASES[int(base_id[i])])
            gi = g.info

            # Construire un set de "mid-params" basé sur base_q (proxy)
            p_mid: dict[str, float | int | bool | str] = {}
            for p in gi.param_specs:
                if p.type in ("float", "int") and p.range:
                    lo, hi = p.range
                    x = float(lo) + (float(hi) - float(lo)) * float(base_q[i, 0, 0].item())
                    p_mid[p.name] = int(round(x)) if p.type == "int" else x
                elif p.type == "enum":
                    p_mid[p.name] = p.enum[0]
                elif p.type == "bool":
                    p_mid[p.name] = False
                else:
                    p_mid[p.name] = 0.0

            # Encode via ParamCodec et rendu 1x
            from ..params import ParamCodec
            P = ParamCodec(gi).to_tensor(p_mid, device=device, dtype=dtype).unsqueeze(0)
            y0 = g.render((h, w), P, seed_base[i:i+1].view(1), device=device, dtype=dtype)[0, 0]
            outs.append(y0)

        y = torch.stack(outs, dim=0).unsqueeze(1)  # [B,1,h,w], ~[-1,1]

        # ------------------------------------------------------------------
        # 2) WARP (optionnel)
        # ------------------------------------------------------------------
        if (warp_id != 0).any():
            amp = (0.02 + 0.18 * warp_q).to(dtype)  # [B,1,1]
            grid_x = xx.clone()  # [B,H,W]
            grid_y = yy.clone()

            # FBM-like (Perlin 2D) -> displacement
            fbm_mask = (warp_id == 1)
            if fbm_mask.any():
                ixs = torch.nonzero(fbm_mask, as_tuple=False).flatten()
                for i in ixs.tolist():
                    sc = (16.0 + 240.0 * float(warp_q[i, 0, 0].item()))
                    s64 = seed_warp[i:i+1].view(1, 1, 1)
                    nx = perlin2d(xx[i:i+1], yy[i:i+1],
                                  torch.tensor([[[sc]]], device=device, dtype=dtype), s64)  # [1,H,W]
                    ny = perlin2d(xx[i:i+1], yy[i:i+1],
                                  torch.tensor([[[sc * 1.37]]], device=device, dtype=dtype),
                                  (s64 ^ _GOLDEN))  # [1,H,W]
                    grid_x[i] = xx[i] + amp[i] * nx[0]
                    grid_y[i] = yy[i] + amp[i] * ny[0]

            # RIPPLE radial
            ripple_mask = (warp_id == 2)
            if ripple_mask.any():
                ixs = torch.nonzero(ripple_mask, as_tuple=False).flatten()
                for i in ixs.tolist():
                    r_i = torch.sqrt(xx[i:i+1] * xx[i:i+1] + yy[i:i+1] * yy[i:i+1] + 1e-6)  # [1,H,W]
                    freq = 4.0 + 12.0 * float(warp_q[i, 0, 0].item())
                    phase = (seed_warp[i, 0, 0].float() % 1000.0) / 1000.0
                    s = torch.sin(2.0 * math.pi * (r_i * freq + phase))  # [1,H,W]
                    c = torch.cos(2.0 * math.pi * (r_i * freq + phase))  # [1,H,W]
                    grid_x[i] = xx[i] + amp[i] * s[0]
                    grid_y[i] = yy[i] + amp[i] * c[0]

            # SWIRL : rotation dépendant du rayon
            swirl_mask = (warp_id == 3)
            if swirl_mask.any():
                ixs = torch.nonzero(swirl_mask, as_tuple=False).flatten()
                for i in ixs.tolist():
                    r_i = torch.sqrt(xx[i:i+1] * xx[i:i+1] + yy[i:i+1] * yy[i:i+1])  # [1,H,W]
                    k = 1.0 + 3.0 * float(warp_q[i, 0, 0].item())
                    theta = k * r_i  # [1,H,W]
                    ct = torch.cos(theta)  # [1,H,W]
                    st = torch.sin(theta)  # [1,H,W]
                    X = xx[i] * ct[0] - yy[i] * st[0]
                    Y = xx[i] * st[0] + yy[i] * ct[0]
                    grid_x[i] = (1.0 - amp[i]) * xx[i] + amp[i] * X
                    grid_y[i] = (1.0 - amp[i]) * yy[i] + amp[i] * Y

            # Applique la warp via grid_sample (bilinear)
            grid_xy = torch.stack([grid_x.clamp(-1, 1), grid_y.clamp(-1, 1)], dim=-1)  # [B,H,W,2]
            y = torch.nn.functional.grid_sample(
                y, grid_xy, mode="bilinear", padding_mode="border", align_corners=True
            )

        # ------------------------------------------------------------------
        # 3) MASK (optionnel) : blend y avec une moyenne en-dehors du masque
        # ------------------------------------------------------------------
        if (mask_id != 0).any():
            y_mean = y.mean(dim=(2, 3), keepdim=True)  # [B,1,1,1]
            sc = (20.0 + 220.0 * mask_q).to(dtype)  # [B,1,1]
            # worley_f1 broadcast : xx[:1]/yy[:1] -> [1,H,W], sc/seed -> [B,1,1] => sortie [B,H,W]
            d = worley_f1(xx[:1], yy[:1], sc, seed_mask.view(B, 1, 1), metric="euclidean")  # [B,H,W]

            mask = torch.zeros((B, 1, h, w), device=device, dtype=dtype)

            ve_mask = (mask_id == 1)  # VORONOI_EDGES
            if ve_mask.any():
                ixs = torch.nonzero(ve_mask, as_tuple=False).flatten().tolist()
                sharp = 8.0
                m = 1.0 - torch.exp(-sharp * d)   # edges brillants
                mask[ixs] = m[ixs].unsqueeze(1)

            dots_mask = (mask_id == 2)  # DOTS
            if dots_mask.any():
                ixs = torch.nonzero(dots_mask, as_tuple=False).flatten().tolist()
                t = 0.12
                m = 1.0 - _smoothstep(d, 0.0, t)  # spots ~1 au centre
                mask[ixs] = m[ixs].unsqueeze(1)

            hex_mask = (mask_id == 3)  # HEX_GRID (approx soft edges)
            if hex_mask.any():
                ixs = torch.nonzero(hex_mask, as_tuple=False).flatten().tolist()
                m = (1.0 - d).pow(2.0)
                mask[ixs] = m[ixs].unsqueeze(1)

            # Blend
            y = y * mask + (1.0 - mask) * y_mean

        # ------------------------------------------------------------------
        # 4) PALETTE (optionnel) : courbes tonales grises
        # ------------------------------------------------------------------
        if (pal_id != 0).any():
            t = ((y + 1.0) * 0.5).clamp(0.0, 1.0)

            pal2 = (pal_id == 1)
            if pal2.any():
                ixs = torch.nonzero(pal2, as_tuple=False).flatten().tolist()
                gamma = (0.6 + 1.4 * pal_q).to(dtype)  # [B,1,1]
                t[ixs] = t[ixs].pow(gamma[ixs])

            pal3 = (pal_id == 2)
            if pal3.any():
                ixs = torch.nonzero(pal3, as_tuple=False).flatten().tolist()
                a = (0.3 + 0.4 * pal_q).to(dtype)  # [B,1,1]
                t[ixs] = (1 - a[ixs]) * t[ixs] + a[ixs] * (t[ixs] * (2 - t[ixs]))

            y = (t * 2.0 - 1.0).clamp(-1.0, 1.0)

        return y.clamp(-1, 1)


GEN = Composite()
