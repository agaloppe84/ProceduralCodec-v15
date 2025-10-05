from __future__ import annotations
import torch
from ..api import Generator, GeneratorInfo, ParamSpec
from ..utils import grid
from ..noise import perlin2d

class CheckerDistorted(Generator):
    @property
    def info(self) -> GeneratorInfo:
        return GeneratorInfo(
            name="CHECKER_DISTORTED",
            param_specs=(
                ParamSpec("cells", "int", (2, 256), "per img", 2.0),
                ParamSpec("dist_scale", "float", (1.0, 64.0), "cells/img", 1.0),
                ParamSpec("dist_amp", "float", (0.0, 0.5), None, 0.01),
            ),
            supports_noise=True,
        )
    @torch.no_grad()
    def render(self, tiles_hw, params, seeds, *, device, dtype):
        B=params.shape[0]; h,w=tiles_hw
        xx,yy = grid(h,w,device=device,dtype=dtype); xx=xx.unsqueeze(0); yy=yy.unsqueeze(0)
        cells=params[:,0].view(B,1,1).to(dtype)
        ds=params[:,1].view(B,1,1).to(dtype)
        da=params[:,2].view(B,1,1).to(dtype)
        seed64=seeds.view(B,1,1).to(torch.int64)
        nx = perlin2d(xx, yy, ds, seed64)
        ny = perlin2d(xx+10, yy-7, ds, seed64^1234)
        xx2 = (xx + da*nx).clamp(-1,1)
        yy2 = (yy + da*ny).clamp(-1,1)
        fx=(xx2+1.0)*cells*0.5; fy=(yy2+1.0)*cells*0.5
        xi=torch.floor(fx).to(torch.int64); yi=torch.floor(fy).to(torch.int64)
        v = ((xi + yi) & 1).to(dtype)*2.0 - 1.0
        return v.unsqueeze(1)

GEN = CheckerDistorted()
