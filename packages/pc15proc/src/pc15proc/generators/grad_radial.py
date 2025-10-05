from __future__ import annotations
import torch
from .api import Generator, GeneratorInfo, ParamSpec
from .utils import grid

class GradRadial(Generator):
    @property
    def info(self) -> GeneratorInfo:
        return GeneratorInfo(
            name="GRADIENT_RADIAL",
            param_specs=(
                ParamSpec("cx", "float", (-1.0, 1.0), None, 0.1),
                ParamSpec("cy", "float", (-1.0, 1.0), None, 0.1),
                ParamSpec("scale", "float", (0.1, 3.0), None, 0.1),
            ),
            supports_noise=False,
        )

    @torch.no_grad()
    def render(self, tiles_hw, params, seeds, *, device, dtype):
        B = params.shape[0]
        h,w = tiles_hw
        xx,yy = grid(h,w,device=device,dtype=dtype)
        xx=xx.unsqueeze(0); yy=yy.unsqueeze(0)
        cx = params[:,0].view(B,1,1).to(dtype)
        cy = params[:,1].view(B,1,1).to(dtype)
        sc = params[:,2].view(B,1,1).to(dtype)
        r = torch.sqrt((xx-cx)**2 + (yy-cy)**2)*sc
        out = (1.0 - r).clamp(-1,1)
        return out.unsqueeze(1)

GEN = GradRadial()
