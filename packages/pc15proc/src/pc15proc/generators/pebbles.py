from __future__ import annotations
import torch
from ..api import Generator, GeneratorInfo, ParamSpec
from ..utils import grid
from ..noise import _hash2, rand01

class Pebbles(Generator):
    @property
    def info(self) -> GeneratorInfo:
        return GeneratorInfo(
            name="PEBBLES",
            param_specs=(
                ParamSpec("scale", "float", (2.0, 256.0), "cells/img", 1.0),
                ParamSpec("smooth", "float", (0.5, 5.0), None, 0.1),
            ),
            supports_noise=True,
        )
    @torch.no_grad()
    def render(self, tiles_hw, params, seeds, *, device, dtype):
        B=params.shape[0]; h,w=tiles_hw
        xx,yy = grid(h,w,device=device,dtype=dtype); xx=xx.unsqueeze(0); yy=yy.unsqueeze(0)
        sc=params[:,0].view(B,1,1).to(dtype); sm=params[:,1].view(B,1,1).to(dtype)
        seed64=seeds.view(B,1,1).to(torch.int64)
        fx=(xx+1.0)*(sc/2.0); fy=(yy+1.0)*(sc/2.0)
        xi=torch.floor(fx).to(torch.int64); yi=torch.floor(fy).to(torch.int64)
        xf=fx - xi.to(dtype); yf=fy - yi.to(dtype)
        best_d = torch.full((B,h,w), float("inf"), device=device, dtype=dtype)
        for oy in (-1,0,1):
            for ox in (-1,0,1):
                cx=xi+ox; cy=yi+oy
                h2=_hash2(cx,cy,seed64)
                jx=rand01(h2)-0.5; jy=rand01(h2 ^ torch.tensor(0x9E3779B97F4A7C15, device=device, dtype=torch.int64))-0.5
                px=ox+0.5+jx; py=oy+0.5+jy
                d2=(xf - px)**2 + (yf - py)**2
                best_d=torch.minimum(best_d, d2)
        v = torch.exp(-sm*best_d)
        return (v*2.0 - 1.0).clamp(-1,1).unsqueeze(1)

GEN = Pebbles()
