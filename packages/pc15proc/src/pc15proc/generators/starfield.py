from __future__ import annotations
import torch
from .api import Generator, GeneratorInfo, ParamSpec
from .utils import grid
from .noise import _hash2, rand01

class Starfield(Generator):
    @property
    def info(self) -> GeneratorInfo:
        return GeneratorInfo(
            name="STARFIELD",
            param_specs=(
                ParamSpec("cells", "int", (8, 512), "per img", 8.0),
                ParamSpec("density", "float", (0.05, 1.0), None, 0.05),
                ParamSpec("sigma", "float", (0.002, 0.05), None, 0.002),
            ),
            supports_noise=True,
        )
    @torch.no_grad()
    def render(self, tiles_hw, params, seeds, *, device, dtype):
        B=params.shape[0]; h,w=tiles_hw
        xx,yy=grid(h,w,device=device,dtype=dtype); xx=xx.unsqueeze(0); yy=yy.unsqueeze(0)
        cells=params[:,0].view(B,1,1).to(dtype)
        dens=params[:,1].view(B,1,1).to(dtype)
        sigma=params[:,2].view(B,1,1).to(dtype)
        seed64=seeds.view(B,1,1).to(torch.int64)
        fx=(xx+1.0)*cells*0.5; fy=(yy+1.0)*cells*0.5
        xi=torch.floor(fx).to(torch.int64); yi=torch.floor(fy).to(torch.int64)
        xf=fx - xi.to(dtype); yf=fy - yi.to(dtype)
        out=torch.full((B,h,w), -1.0, device=device, dtype=dtype)
        for oy in (-1,0,1):
            for ox in (-1,0,1):
                cx=xi+ox; cy=yi+oy
                h2=_hash2(cx,cy,seed64)
                keep = rand01(h2) < dens
                jx = rand01(h2) - 0.5
                jy = rand01(h2 ^ torch.tensor(0x9E3779B97F4A7C15, device=device, dtype=torch.int64)) - 0.5
                px = ox + 0.5 + jx; py = oy + 0.5 + jy
                d2 = (xf - px)**2 + (yf - py)**2
                g = torch.exp(-0.5 * d2 / (sigma*sigma))
                out = torch.maximum(out, (g*2.0 - 1.0) * keep.to(dtype) + (-1.0)*(~keep).to(dtype))
        return out.unsqueeze(1)

GEN = Starfield()
