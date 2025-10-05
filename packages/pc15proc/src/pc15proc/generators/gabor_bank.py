from __future__ import annotations
import torch
from .api import Generator, GeneratorInfo, ParamSpec
from .utils import grid

class GaborBank(Generator):
    @property
    def info(self) -> GeneratorInfo:
        return GeneratorInfo(
            name="GABOR_BANK",
            param_specs=(
                ParamSpec("freq", "float", (0.5, 64.0), "cycles/img", 0.5),
                ParamSpec("sigma", "float", (0.05, 1.0), None, 0.05),
                ParamSpec("bands", "int", (1, 6), None, 1.0),
            ),
            supports_noise=False,
        )
    @torch.no_grad()
    def render(self, tiles_hw, params, seeds, *, device, dtype):
        B=params.shape[0]; h,w=tiles_hw
        xx,yy=grid(h,w,device=device,dtype=dtype); xx=xx.unsqueeze(0); yy=yy.unsqueeze(0)
        f=params[:,0].view(B,1,1).to(dtype)
        sig=params[:,1].view(B,1,1).to(dtype)
        bands=params[:,2].view(B,1,1).to(torch.int64)
        out=torch.zeros((B,h,w), device=device, dtype=dtype)
        for k in range(6):
            use=(k<bands).to(dtype)
            ang = (torch.pi * k / (bands.to(dtype)*2.0 + 1e-6)).view(B,1,1)
            xr = xx*torch.cos(ang) + yy*torch.sin(ang)
            yr = -xx*torch.sin(ang) + yy*torch.cos(ang)
            env = torch.exp(-0.5*((xr/sig)**2 + (yr/sig)**2))
            s = torch.sin(2*torch.pi*f*xr) * env
            out = out + s * use
        out = out / (bands.to(dtype) + 1e-6)
        return out.clamp(-1,1).unsqueeze(1)

GEN = GaborBank()
