from __future__ import annotations
import torch
from .api import Generator, GeneratorInfo, ParamSpec
from .utils import grid
from .noise import _hash2, rand01

class VoronoiCells(Generator):
    @property
    def info(self) -> GeneratorInfo:
        return GeneratorInfo(
            name="VORONOI_CELLS",
            param_specs=(
                ParamSpec("scale", "float", (1.0, 256.0), "cells/img", 1.0),
                ParamSpec("contrast", "float", (0.5, 2.0), None, 0.1),
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
                jx=rand01(h2); jy=rand01(h2 ^ torch.tensor(0x9E3779B97F4A7C15, device=device, dtype=torch.int64))
                px=ox + jx; py=oy + jy
                dx=xf - px; dy=yf - py
                d2=dx*dx + dy*dy
                better = d2 < best_d
                best_d = torch.where(better, d2, best_d)
                best_ix = torch.where(better, cx, best_ix)
                best_iy = torch.where(better, cy, best_iy)

        hcell = _hash2(best_ix, best_iy, seed64)
        val = rand01(hcell)  # [0,1)
        val = ( (val - 0.5) * c + 0.5 ).clamp(0,1)
        out = val*2.0 - 1.0
        return out.unsqueeze(1)

GEN = VoronoiCells()
