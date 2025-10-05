from __future__ import annotations
import torch
from ..api import Generator, GeneratorInfo, ParamSpec
from ..utils import grid

class WavesCirc(Generator):
    @property
    def info(self) -> GeneratorInfo:
        return GeneratorInfo(
            name="WAVES_CIRC",
            param_specs=(
                ParamSpec("freq", "float", (0.5, 128.0), "cycles/img", 0.5),
                ParamSpec("amp", "float", (0.1, 1.0), None, 0.1),
                ParamSpec("phase", "float", (0.0, 6.28318), "rad", 0.2),
            ),
            supports_noise=False,
        )

    @torch.no_grad()
    def render(self, tiles_hw, params, seeds, *, device, dtype):
        B=params.shape[0]
        h,w=tiles_hw
        xx,yy=grid(h,w,device=device,dtype=dtype)
        xx=xx.unsqueeze(0); yy=yy.unsqueeze(0)
        f=params[:,0].view(B,1,1).to(dtype)
        a=params[:,1].view(B,1,1).to(dtype)
        ph=params[:,2].view(B,1,1).to(dtype)
        r=torch.sqrt(xx*xx+yy*yy)
        s=torch.sin(2*torch.pi*f*r + ph)*a
        return s.clamp(-1,1).unsqueeze(1)

GEN = WavesCirc()
