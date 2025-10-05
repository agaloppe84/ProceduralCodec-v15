from __future__ import annotations
import torch
from .api import Generator, GeneratorInfo, ParamSpec
from .utils import grid
from .noise import _hash2, rand01

class VoronoiEdges(Generator):
    @property
    def info(self) -> GeneratorInfo:
        return GeneratorInfo(
            name="VORONOI_EDGES",
            param_specs=(
                ParamSpec("scale", "float", (1.0, 256.0), "cells/img", 1.0),
                ParamSpec("sharp", "float", (0.5, 10.0), None, 0.5),
            ),
            supports_noise=True,
        )

    @torch.no_grad()
    def render(self, tiles_hw, params, seeds, *, device, dtype):
        B=params.shape[0]; h,w=tiles_hw
        xx,yy=grid(h,w,device=device,dtype=dtype); xx=xx.unsqueeze(0); yy=yy.unsqueeze(0)
        sc=params[:,0].view(B,1,1).to(dtype)
        sharp=params[:,1].view(B,1,1).to(dtype)
        seed64=seeds.view(B,1,1).to(torch.int64)

        fx=(xx+1.0)*(sc/2.0)
        fy=(yy+1.0)*(sc/2.0)
        xi=torch.floor(fx).to(torch.int64)
        yi=torch.floor(fy).to(torch.int64)
        xf=fx - xi.to(dtype)
        yf=fy - yi.to(dtype)

        ds=[]
        for oy in (-1,0,1):
            for ox in (-1,0,1):
                cx=xi+ox; cy=yi+oy
                h2=_hash2(cx,cy,seed64)
                jx=rand01(h2); jy=rand01(h2 ^ torch.tensor(0x9E3779B97F4A7C15, device=device, dtype=torch.int64))
                px=ox + jx; py=oy + jy
                dx=xf - px; dy=yf - py
                d=torch.sqrt(dx*dx + dy*dy)
                ds.append(d)
        D=torch.stack(ds, dim=0)  # [9,B,h,w]
        d_sorted, _ = torch.sort(D, dim=0)
        f1 = d_sorted[0]
        f2 = d_sorted[1]
        e = torch.exp(-sharp * (f2 - f1))
        out = (1.0 - e)*2.0 - 1.0
        return out.clamp(-1,1).unsqueeze(1)

GEN = VoronoiEdges()
