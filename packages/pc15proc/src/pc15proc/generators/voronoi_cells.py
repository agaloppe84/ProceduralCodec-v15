from __future__ import annotations
import torch
from ..api import Generator, GeneratorInfo, ParamSpec
from ..utils import grid
from ..noise import _hash2, rand01
from pc15core.rng import to_int64_signed   # ⬅️ AJOUT

# u64 → int64 signé (mêmes bits), réutilisé pour l’XOR
_GOLDEN64_I = to_int64_signed(0x9E3779B97F4A7C15)   # ⬅️ AJOUT

class VoronoiCells(Generator):
    @property
    def info(self) -> GeneratorInfo:
        return GeneratorInfo(
            name="VORONOI_CELLS",
            param_specs=(
                ParamSpec("scale", "int", (16, 256), "per img", 64.0),
                ParamSpec("contrast", "float", (0.5, 5.0), None, 1.0),
            ),
            supports_noise=True,
        )

    @torch.no_grad()
    def render(self, tiles_hw, params, seeds, *, device, dtype):
        B=params.shape[0]; h,w=tiles_hw
        xx,yy=grid(h,w,device=device,dtype=dtype); xx=xx.unsqueeze(0); yy=yy.unsqueeze(0)
        sc=params[:,0].view(B,1,1).to(dtype)
        c =params[:,1].view(B,1,1).to(dtype)
        seed64=seeds.view(B,1,1).to(torch.int64)
        fx=(xx+1.0)*(sc/2.0); fy=(yy+1.0)*(sc/2.0)
        xi=torch.floor(fx).to(torch.int64); yi=torch.floor(fy).to(torch.int64)
        xf=fx - xi.to(dtype); yf=fy - yi.to(dtype)

        best_d = torch.full((B,h,w), float("inf"), device=device, dtype=dtype)
        best_ix = torch.zeros((B,h,w), device=device, dtype=torch.int64)
        best_iy = torch.zeros((B,h,w), device=device, dtype=torch.int64)

        for oy in (-1,0,1):
            for ox in (-1,0,1):
                cx=xi+ox; cy=yi+oy
                h2=_hash2(cx,cy,seed64)
                jx=rand01(h2)
                jy=rand01(h2 ^ _GOLDEN64_I)   # ⬅️ remplacé: plus de torch.tensor(...)

                dx=(cx.to(dtype)+jx)-fx
                dy=(cy.to(dtype)+jy)-fy
                d2=dx*dx + dy*dy
                mask = d2 < best_d
                best_d = torch.where(mask, d2, best_d)
                best_ix = torch.where(mask, cx, best_ix)
                best_iy = torch.where(mask, cy, best_iy)

        # sortie simple basée sur la distance au site le plus proche
        out = torch.exp(-c * best_d.sqrt()).unsqueeze(1) * 2.0 - 1.0
        return out.clamp(-1,1)

GEN = VoronoiCells()
