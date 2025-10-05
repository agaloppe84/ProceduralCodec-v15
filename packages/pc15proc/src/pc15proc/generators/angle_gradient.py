from __future__ import annotations
import torch
from .api import Generator, GeneratorInfo, ParamSpec
from .utils import grid

class AngleGradient(Generator):
    @property
    def info(self) -> GeneratorInfo:
        return GeneratorInfo(
            name="ANGLE_GRADIENT",
            param_specs=(
                ParamSpec("angle_start_deg", "float", (0.0, 360.0), "deg", 5.0),
                ParamSpec("angle_end_deg", "float", (0.0, 360.0), "deg", 5.0),
            ),
            supports_noise=False,
        )
    @torch.no_grad()
    def render(self, tiles_hw, params, seeds, *, device, dtype):
        B = params.shape[0]; h,w=tiles_hw
        xx,yy = grid(h,w,device=device,dtype=dtype); xx=xx.unsqueeze(0); yy=yy.unsqueeze(0)
        a0 = torch.deg2rad(params[:,0]).view(B,1,1).to(dtype)
        a1 = torch.deg2rad(params[:,1]).view(B,1,1).to(dtype)
        t = (yy + 1.0) * 0.5  # 0..1 from top->bottom
        ang = a0 * (1.0 - t) + a1 * t
        u = xx*torch.cos(ang) + yy*torch.sin(ang)
        return u.clamp(-1,1).unsqueeze(1)

GEN = AngleGradient()
