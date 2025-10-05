from __future__ import annotations
import torch
from ..api import Generator, GeneratorInfo, ParamSpec
from ..utils import grid
from ..noise import _hash2, rand01

class GaussianSpots(Generator):
    @property
    def info(self) -> GeneratorInfo:
        return GeneratorInfo(
            name="GAUSSIAN_SPOTS",
            param_specs=(
                ParamSpec("cells", "int", (4, 256), "per img", 4.0),
                ParamSpec("sigma", "float", (0.01, 0.2), None, 0.01),
                ParamSpec("density", "float", (0.1, 1.0), None, 0.05),
            ),
            supports_noise=True,
        )

    @torch.no_grad()
    def render(self, tiles_hw, params, seeds, *, device, dtype):
        B=params.shape[0]; h,w=tiles_hw
        xx,yy=grid(h,w,device=device,dtype=dtype); xx=xx.unsqueeze(0); yy=yy.unsqueeze(0)
        cells=params[:,0].view(B,1,1).to(dtype)
        sigma=params[:,1].view(B,1,1).to(dtype)
        dens =params[:,2].view(B,1,1).to(dtype)
        seed64=seeds.view(B,1,1).to(torch.int64)

        fx=(xx+1.0)*cells*0.5
        fy=(yy+1.0)*cells*0.5
        xi=torch.floor(fx).to(torch.int64)
        yi=torch.floor(fy).to(torch.int64)
        xf=fx - xi.to(dtype)
        yf=fy - yi.to(dtype)

        out=torch.zeros((B,h,w), device=device, dtype=dtype)
        for oy in (-1,0,1):
            for ox in (-1,0,1):
                cx = xi + ox
                cy = yi + oy
                h2 = _hash2(cx, cy, seed64)
                choose = rand01(h2) < dens
                jx = rand01(h2) - 0.5
                jy = rand01(h2 ^ torch.tensor(0x9E3779B97F4A7C15, device=device, dtype=torch.int64)) - 0.5
                px = ox + 0.5 + jx
                py = oy + 0.5 + jy
                dx = xf - px
                dy = yf - py
                d2 = dx*dx + dy*dy
                g = torch.exp(-0.5 * (d2 / (sigma*sigma)))
                out = out + g * choose.to(dtype)
        out = (out / out.amax(dim=(1,2), keepdim=True).clamp(min=1e-6))*2.0 - 1.0
        return out.clamp(-1,1).unsqueeze(1)

GEN = GaussianSpots()
