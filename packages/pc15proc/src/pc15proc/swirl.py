from __future__ import annotations
import torch
from .api import Generator, GeneratorInfo, ParamSpec
from .utils import grid

class Swirl(Generator):
    @property
    def info(self) -> GeneratorInfo:
        return GeneratorInfo(
            name="SWIRL",
            param_specs=(
                ParamSpec("k", "float", (0.0, 10.0), None, 0.1),
                ParamSpec("freq", "float", (0.5, 64.0), "cycles/img", 0.5),
            ),
            supports_noise=False,
        )
    @torch.no_grad()
    def render(self, tiles_hw, params, seeds, *, device, dtype):
        B=params.shape[0]; h,w=tiles_hw
        xx,yy=grid(h,w,device=device,dtype=dtype); xx=xx.unsqueeze(0); yy=yy.unsqueeze(0)
        k=params[:,0].view(B,1,1).to(dtype); f=params[:,1].view(B,1,1).to(dtype)
        r2=xx*xx+yy*yy
        ang = k * r2
        xr =  xx*torch.cos(ang) - yy*torch.sin(ang)
        s = torch.sin(2*torch.pi*f*xr)
        return s.clamp(-1,1).unsqueeze(1)

GEN = Swirl()
