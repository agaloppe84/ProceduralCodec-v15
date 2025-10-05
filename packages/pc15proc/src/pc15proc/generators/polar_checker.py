from __future__ import annotations
import torch
from .api import Generator, GeneratorInfo, ParamSpec
from .utils import grid

class PolarChecker(Generator):
    @property
    def info(self) -> GeneratorInfo:
        return GeneratorInfo(
            name="POLAR_CHECKER",
            param_specs=(
                ParamSpec("radial", "int", (2, 256), "rings", 2.0),
                ParamSpec("angular", "int", (2, 256), "sectors", 2.0),
            ),
            supports_noise=False,
        )
    @torch.no_grad()
    def render(self, tiles_hw, params, seeds, *, device, dtype):
        B=params.shape[0]; h,w=tiles_hw
        xx,yy=grid(h,w,device=device,dtype=dtype); xx=xx.unsqueeze(0); yy=yy.unsqueeze(0)
        rcount=params[:,0].view(B,1,1).to(dtype)
        acount=params[:,1].view(B,1,1).to(dtype)
        r = torch.sqrt(xx*xx + yy*yy)
        ring = torch.floor((r.clamp(0,1)) * rcount)
        theta = (torch.atan2(yy,xx) + torch.pi) / (2*torch.pi)
        sector = torch.floor(theta * acount)
        v = (((ring + sector) % 2)==0).to(dtype)*2.0 - 1.0
        return v.unsqueeze(1)

GEN = PolarChecker()
