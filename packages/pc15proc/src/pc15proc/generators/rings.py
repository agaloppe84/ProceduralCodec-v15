from __future__ import annotations
import torch
from ..api import Generator, GeneratorInfo, ParamSpec
from ..utils import grid

class Rings(Generator):
    @property
    def info(self) -> GeneratorInfo:
        return GeneratorInfo(
            name="RINGS",
            param_specs=(
                ParamSpec("freq", "float", (0.5, 128.0), "cycles/img", 0.5),
                ParamSpec("cx", "float", (-1.0, 1.0), None, 0.1),
                ParamSpec("cy", "float", (-1.0, 1.0), None, 0.1),
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
        freq=params[:,0].view(B,1,1).to(dtype)
        cx=params[:,1].view(B,1,1).to(dtype)
        cy=params[:,2].view(B,1,1).to(dtype)
        ph=params[:,3].view(B,1,1).to(dtype)
        r=torch.sqrt((xx-cx)**2+(yy-cy)**2)
        s=torch.sin(2*torch.pi*freq*r+ph)
        return s.clamp(-1,1).unsqueeze(1)

GEN = Rings()
