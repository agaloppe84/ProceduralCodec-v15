from __future__ import annotations
import torch
from .api import Generator, GeneratorInfo, ParamSpec
from .utils import grid
from .noise import perlin2d

class GaborNoise(Generator):
    @property
    def info(self) -> GeneratorInfo:
        return GeneratorInfo(
            name="GABOR_NOISE",
            param_specs=(
                ParamSpec("freq", "float", (0.5, 64.0), "cycles/img", 0.5),
                ParamSpec("sigma", "float", (0.05, 1.0), None, 0.05),
                ParamSpec("dist_scale", "float", (1.0, 64.0), "cells/img", 1.0),
                ParamSpec("dist_amp", "float", (0.0, 0.5), None, 0.01),
            ),
            supports_noise=True,
        )
    @torch.no_grad()
    def render(self, tiles_hw, params, seeds, *, device, dtype):
        B = params.shape[0]; h,w = tiles_hw
        xx,yy = grid(h,w,device=device,dtype=dtype); xx=xx.unsqueeze(0); yy=yy.unsqueeze(0)
        f=params[:,0].view(B,1,1).to(dtype)
        sig=params[:,1].view(B,1,1).to(dtype)
        ds=params[:,2].view(B,1,1).to(dtype)
        da=params[:,3].view(B,1,1).to(dtype)
        seed64=seeds.view(B,1,1).to(torch.int64)
        nx = perlin2d(xx, yy, ds, seed64)
        ny = perlin2d(xx+3, yy-5, ds, seed64^5678)
        xr = (xx + da*nx).clamp(-1,1)
        yr = (yy + da*ny).clamp(-1,1)
        env = torch.exp(-0.5*((xr/sig)**2 + (yr/sig)**2))
        s = torch.sin(2*torch.pi*f*xr) * env
        return s.clamp(-1,1).unsqueeze(1)

GEN = GaborNoise()
