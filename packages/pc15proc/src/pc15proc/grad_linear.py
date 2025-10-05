from __future__ import annotations
import torch
from .api import Generator, GeneratorInfo, ParamSpec
from .utils import grid

class GradLinear(Generator):
    @property
    def info(self) -> GeneratorInfo:
        return GeneratorInfo(
            name="GRADIENT_LINEAR",
            param_specs=(
                ParamSpec("angle_deg", "float", (0.0, 180.0), "deg", 5.0),
                ParamSpec("bias", "float", (-1.0, 1.0), None, 0.1),
                ParamSpec("scale", "float", (0.1, 2.0), None, 0.1),
            ),
            supports_noise=False,
        )

    @torch.no_grad()
    def render(self, tiles_hw, params, seeds, *, device, dtype):
        B = params.shape[0]
        h,w = tiles_hw
        xx,yy = grid(h,w,device=device,dtype=dtype)
        xx=xx.unsqueeze(0); yy=yy.unsqueeze(0)
        ang  = torch.deg2rad(params[:,0]).view(B,1,1).to(dtype)
        bias = params[:,1].view(B,1,1).to(dtype)
        scale= params[:,2].view(B,1,1).to(dtype)
        u = (xx*torch.cos(ang) + yy*torch.sin(ang))*scale + bias
        return u.clamp(-1,1).unsqueeze(1)

GEN = GradLinear()
