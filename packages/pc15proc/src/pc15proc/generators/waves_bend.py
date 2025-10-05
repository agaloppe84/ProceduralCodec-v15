from __future__ import annotations
import torch
from .api import Generator, GeneratorInfo, ParamSpec
from .utils import grid

class WavesBend(Generator):
    @property
    def info(self) -> GeneratorInfo:
        return GeneratorInfo(
            name="WAVES_BEND",
            param_specs=(
                ParamSpec("freq", "float", (0.5, 64.0), "cycles/img", 0.5),
                ParamSpec("bend", "float", (0.0, 0.5), None, 0.01),
                ParamSpec("bend_freq", "float", (0.5, 32.0), "cycles/img", 0.5),
            ),
            supports_noise=False,
        )
    @torch.no_grad()
    def render(self, tiles_hw, params, seeds, *, device, dtype):
        B=params.shape[0]; h,w=tiles_hw
        xx,yy=grid(h,w,device=device,dtype=dtype); xx=xx.unsqueeze(0); yy=yy.unsqueeze(0)
        f =params[:,0].view(B,1,1).to(dtype)
        b =params[:,1].view(B,1,1).to(dtype)
        bf=params[:,2].view(B,1,1).to(dtype)
        u = xx + b * torch.sin(2*torch.pi*bf*yy)
        s = torch.sin(2*torch.pi*f*u)
        return s.clamp(-1,1).unsqueeze(1)

GEN = WavesBend()
