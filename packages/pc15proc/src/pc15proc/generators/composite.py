from __future__ import annotations
import math
import torch

from ..api import Generator, GeneratorInfo, ParamSpec
from ..utils import grid
from ..noise import perlin2d, worley_f1
from ..registry import get as get_gen
from ..params import ParamCodec
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

        # Coordonnées normalisées pour grid_sample
        xx, yy = grid(h, w, device=device, dtype=dtype)  # [1,H,W] in [-1,1]
        xx = xx.expand(B, h, w).contiguous()             # [B,H,W]
        yy = yy.expand(B, h, w).contiguous()             # [B,H,W]
        r  = torch.sqrt(xx * xx + yy * yy + 1e-6)        # [B,H,W] (réutilisé)

        # Decode params (shape-safe)
        base_id = params[:, 0].round().clamp(0, len(_BASES) - 1).to(torch.int64)
        warp_id = params[:, 1].round().clamp(0, len(_WARPS) - 1).to(torch.int64)
        mask_id = params[:, 2].round().clamp(0, len(_MASKS) - 1).to(torch.int64)
        pal_id  = params[:, 3].round().clamp(0, len(_PALS)  - 1).to(torch.int64)

        base_q = params[:, 4].to(dtype).view(B, 1, 1)   # [B,1,1]
        warp_q = params[:, 5].to(dtype).view(B, 1, 1)   # [B,1,1]
        mask_q = params[:, 6].to(dtype).view(B, 1, 1)   # [B,1,1]
        pal_q  = params[:, 7].to(dtype).view(B, 1, 1)   # [B,1,1]

        # Seeds dérivés
        seed_tile = seeds.view(B, 1, 1).to(torch.int64)
        seed_base = _slot_seed(seed_tile, 1)            # [B,1,1]
        seed_warp = _slot_seed(seed_tile, 2)            # [B,1,1]
        seed_mask = _slot_seed(seed_tile, 3)            # [B,1,1]
        # seed_pal  = _slot_seed(seed_tile, 4)          # (pas utilisé, palette purement déterministe)

        # ------------------------------------------------------------------
        # 1) BASE : rendre par groupes de base_id (batch par générateur)
        # ------------------------------------------------------------------
        y = torch.empty((B, 1, h, w), device=device, dtype=dtype)

        # cache ParamCodec par type de base
        pc_cache: dict[str, ParamCodec] = {}

        uniq, inv = torch.unique(base_id, sorted=True, return_inverse=True)
        # inv : pour chaque sample b, l'index de uniq correspondant (pas utilisé ici directement)

        for bid in uniq.tolist():
            mask = (base_id == bid)
            ixs = torch.nonzero(mask, as_tuple=False).flatten()
            if ixs.numel() == 0:
                continue

            g = get_gen(_BASES[bid])
            gi = g.info
            if _BASES[bid] not in pc_cache:
                pc_cache[_BASES[bid]] = ParamCodec(gi)
            pc = pc_cache[_BASES[bid]]

            # Construire P_group en fonction de base_q (proxy 0..1) — un dict par sample
            Ps = []
            for i in ixs.tolist():
                q = float(base_q[i, 0, 0].item())
                p_mid: dict[str, float | int | bool | str] = {}
                for p in gi.param_specs:
                    if p.type in ("float", "int") and p.range:
                        lo, hi = p.range
                        x = float(lo) + (float(hi) - float(lo)) * q
                        p_mid[p.name] = int(round(x)) if p.type == "int" else x
                    elif p.type == "enum":
                        p_mid[p.name] = p.enum[0]
                    elif p.type == "bool":
                        p_mid[p.name] = False
                    else:
                        p_mid[p.name] = 0.0
                Ps.append(pc.to_tensor(p_mid, device=device, dtype=dtype))

            P_group = torch.stack(Ps, dim=0)  # [Ng, D]
            seeds_group = seed_base[ixs].view(-1)  # [Ng] int64

            y_group = g.render((h, w), P_group, seeds_group, device=device, dtype=dtype)  # [Ng,1,H,W]
            y[ixs] = y_group

        # ------------------------------------------------------------------
        # 2) WARP (optionnel) — vectorisé par groupes
        # ------------------------------------------------------------------
        if (warp_id != 0).any():
            amp = (0.02 + 0.18 * warp_q).to(dtype)                 # [B,1,1]
            grid_x = xx.clone()                                     # [B,H,W]
            grid_y = yy.clone()                                     # [B,H,W]

            # FBM (Perlin 2D)
            fbm_mask = (warp_id == 1)
            if fbm_mask.any():
                ixs = torch.nonzero(fbm_mask, as_tuple=False).flatten()
                xx_g = xx[ixs]                                      # [Ng,H,W]
                yy_g = yy[ixs]
                sc_g = (16.0 + 240.0 * warp_q[ixs]).to(dtype)       # [Ng,1,1]
                s64  = seed_warp[ixs].view(-1, 1, 1)                # [Ng,1,1]
                nx = perlin2d(xx_g, yy_g, sc_g, s64)                # [Ng,H,W]
                ny = perlin2d(xx_g, yy_g, sc_g * 1.37, s64 ^ _GOLDEN)
                grid_x[ixs] = xx_g + amp[ixs] * nx
                grid_y[ixs] = yy_g + amp[ixs] * ny

            # RIPPLE radial
            ripple_mask = (warp_id == 2)
            if ripple_mask.any():
                ixs = torch.nonzero(ripple_mask, as_tuple=False).flatten()
                r_g = r[ixs]                                        # [Ng,H,W]
                freq = (4.0 + 12.0 * warp_q[ixs]).to(dtype)         # [Ng,1,1]
                phase = (seed_warp[ixs].to(torch.float32) % 1000.0) / 1000.0
                phase = phase.view(-1, 1, 1).to(dtype)              # [Ng,1,1]
                arg = r_g * freq + phase                            # [Ng,H,W]
                s = torch.sin(2.0 * math.pi * arg)
                c = torch.cos(2.0 * math.pi * arg)
                grid_x[ixs] = xx[ixs] + amp[ixs] * s
                grid_y[ixs] = yy[ixs] + amp[ixs] * c

            # SWIRL : rotation dépendante du rayon
            swirl_mask = (warp_id == 3)
            if swirl_mask.any():
                ixs = torch.nonzero(swirl_mask, as_tuple=False).flatten()
                r_g = r[ixs]                                        # [Ng,H,W]
                k = (1.0 + 3.0 * warp_q[ixs]).to(dtype)             # [Ng,1,1]
                theta = k * r_g                                     # [Ng,H,W]
                ct = torch.cos(theta)
                st = torch.sin(theta)
                X = xx[ixs] * ct - yy[ixs] * st
                Y = xx[ixs] * st + yy[ixs] * ct
                grid_x[ixs] = (1.0 - amp[ixs]) * xx[ixs] + amp[ixs] * X
                grid_y[ixs] = (1.0 - amp[ixs]) * yy[ixs] + amp[ixs] * Y

            grid_xy = torch.stack([grid_x.clamp(-1, 1), grid_y.clamp(-1, 1)], dim=-1)  # [B,H,W,2]
            y = torch.nn.functional.grid_sample(
                y, grid_xy, mode="bilinear", padding_mode="border", align_corners=True
            )

        # ------------------------------------------------------------------
        # 3) MASK (optionnel) : blend y avec y_mean en dehors du masque
        # ------------------------------------------------------------------
        if (mask_id != 0).any():
            y_mean = y.mean(dim=(2, 3), keepdim=True)               # [B,1,1,1]
            sc = (20.0 + 220.0 * mask_q).to(dtype)                  # [B,1,1]
            # worley_f1 broadcast: xx[:1]/yy[:1] -> [1,H,W]; sc/seed -> [B,1,1] => [B,H,W]
            d = worley_f1(xx[:1], yy[:1], sc, seed_mask.view(B, 1, 1), metric="euclidean")  # [B,H,W]

            mask = torch.zeros((B, 1, h, w), device=device, dtype=dtype)

            # VORONOI_EDGES
            ve_mask = (mask_id == 1)
            if ve_mask.any():
                ixs = torch.nonzero(ve_mask, as_tuple=False).flatten()
                sharp = 8.0
                m = 1.0 - torch.exp(-sharp * d)                      # [B,H,W]
                mask[ixs] = m[ixs].unsqueeze(1)

            # DOTS
            dots_mask = (mask_id == 2)
            if dots_mask.any():
                ixs = torch.nonzero(dots_mask, as_tuple=False).flatten()
                t0 = 0.12
                m = 1.0 - _smoothstep(d, 0.0, t0)                    # [B,H,W]
                mask[ixs] = m[ixs].unsqueeze(1)

            # HEX_GRID (approx soft edges)
            hex_mask = (mask_id == 3)
            if hex_mask.any():
                ixs = torch.nonzero(hex_mask, as_tuple=False).flatten()
                m = (1.0 - d).pow(2.0)
                mask[ixs] = m[ixs].unsqueeze(1)

            y = y * mask + (1.0 - mask) * y_mean

        # ------------------------------------------------------------------
        # 4) PALETTE (optionnel) : courbes tonales grises (broadcast sûr)
        # ------------------------------------------------------------------
        if (pal_id != 0).any():
            # Remap [-1,1] -> [0,1]
            t = ((y + 1.0) * 0.5).clamp(0.0, 1.0)                   # [B,1,H,W]

            pal2_mask = (pal_id == 1).view(B, 1, 1, 1)
            pal3_mask = (pal_id == 2).view(B, 1, 1, 1)

            if pal2_mask.any():
                gamma4 = (0.6 + 1.4 * pal_q.to(dtype)).view(B, 1, 1, 1)  # [B,1,1,1]
                t_pal2 = t.pow(gamma4)
                t = torch.where(pal2_mask, t_pal2, t)

            if pal3_mask.any():
                a4 = (0.3 + 0.4 * pal_q.to(dtype)).view(B, 1, 1, 1)      # [B,1,1,1]
                t_s = (1.0 - a4) * t + a4 * (t * (2.0 - t))              # S-curve douce
                t = torch.where(pal3_mask, t_s, t)

            y = (t * 2.0 - 1.0).clamp(-1.0, 1.0)

        return y.clamp(-1, 1)


GEN = Composite()
