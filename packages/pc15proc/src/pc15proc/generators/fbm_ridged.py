from __future__ import annotations
import torch
from .api import Generator, GeneratorInfo, ParamSpec
from .utils import grid
from .noise import perlin2d

class FBMRidged(Generator):
    @property
    def info(self) -> GeneratorInfo:
        return GeneratorInfo(
            name="FBM_RIDGED",
            param_specs=(
                ParamSpec("scale", "float", (1.0, 256.0), "cells/img", 1.0),
                ParamSpec("octaves", "int", (1, 6), None, 1.0),
                ParamSpec("gain", "float", (0.3, 1.0), None, 0.1),
            ),
            supports_noise=True,
        )
    @torch.no_grad()
    def render(self, tiles_hw, params, seeds, *, device, dtype):
        B=params.shape[0]; h,w=tiles_hw
        xx,yy=grid(h,w,device=device,dtype=dtype); xx=xx.unsqueeze(0); yy=yy.unsqueeze(0)
        base=params[:,0].view(B,1,1).to(dtype)
        octv=params[:,1].view(B,1,1).to(torch.int64)
        gain=params[:,2].view(B,1,1).to(dtype)
        seed64=seeds.view(B,1,1).to(torch.int64)
        val=torch.zeros((B,h,w), device=device, dtype=dtype)
        amp=torch.ones((B,1,1), device=device, dtype=dtype)
        freq=base.clone()
        for k in range(6):
            use=(k<octv).to(dtype)
            n=perlin2d(xx,yy,freq,seed64+k)
            rid = 1.0 - torch.abs(n)
            val = val + rid * amp * use
            amp = amp * gain
            freq = freq * 2.0
        val = val / (1e-6 + (1.0 - torch.pow(gain, octv.to(dtype))) / (1.0 - gain + 1e-6))
        return (val*2.0 - 1.0).clamp(-1,1).unsqueeze(1)

GEN = FBMRidged()
