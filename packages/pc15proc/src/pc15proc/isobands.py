from __future__ import annotations
import torch
from .api import Generator, GeneratorInfo, ParamSpec
from .utils import grid
from .noise import perlin2d

class Isobands(Generator):
    @property
    def info(self) -> GeneratorInfo:
        return GeneratorInfo(
            name="ISOBANDS",
            param_specs=(
                ParamSpec("scale", "float", (1.0, 256.0), "cells/img", 1.0),
                ParamSpec("levels", "int", (2, 32), None, 1.0),
            ),
            supports_noise=True,
        )
    @torch.no_grad()
    def render(self, tiles_hw, params, seeds, *, device, dtype):
        B=params.shape[0]; h,w=tiles_hw
        xx,yy = grid(h,w,device=device,dtype=dtype); xx=xx.unsqueeze(0); yy=yy.unsqueeze(0)
        sc=params[:,0].view(B,1,1).to(dtype)
        L =params[:,1].view(B,1,1).to(torch.int64)
        n = perlin2d(xx, yy, sc, seeds.view(B,1,1).to(torch.int64))  # [-1,1]
        v = (n + 1.0) * 0.5  # [0,1]
        out=torch.zeros_like(v)
        for i in range(1, 32):
            use=(i < L).to(dtype)
            thr = (i / L.to(dtype))
            out = out + ( (v > thr).to(dtype) * use )
        out = (out / (L.to(dtype)+1e-6))*2.0 - 1.0
        return out.unsqueeze(1)

GEN = Isobands()
