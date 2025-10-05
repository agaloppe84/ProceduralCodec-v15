from __future__ import annotations
import torch
from .api import Generator, GeneratorInfo, ParamSpec
from .utils import grid

class StripesMultiscale(Generator):
    @property
    def info(self) -> GeneratorInfo:
        return GeneratorInfo(
            name="STRIPES_MULTISCALE",
            param_specs=(
                ParamSpec("f1", "float", (0.5, 64.0), "cycles/img", 0.5),
                ParamSpec("f2", "float", (0.5, 64.0), "cycles/img", 0.5),
                ParamSpec("angle1_deg", "float", (0.0, 180.0), "deg", 5.0),
                ParamSpec("angle2_deg", "float", (0.0, 180.0), "deg", 5.0),
            ),
            supports_noise=False,
        )
    @torch.no_grad()
    def render(self, tiles_hw, params, seeds, *, device, dtype):
        B=params.shape[0]; h,w=tiles_hw
        xx,yy=grid(h,w,device=device,dtype=dtype); xx=xx.unsqueeze(0); yy=yy.unsqueeze(0)
        f1=params[:,0].view(B,1,1).to(dtype); f2=params[:,1].view(B,1,1).to(dtype)
        a1=torch.deg2rad(params[:,2]).view(B,1,1).to(dtype)
        a2=torch.deg2rad(params[:,3]).view(B,1,1).to(dtype)
        u1 = xx*torch.cos(a1)+yy*torch.sin(a1)
        u2 = xx*torch.cos(a2)+yy*torch.sin(a2)
        s = 0.5*(torch.sin(2*torch.pi*f1*u1) + torch.sin(2*torch.pi*f2*u2))
        return s.clamp(-1,1).unsqueeze(1)

GEN = StripesMultiscale()
