from __future__ import annotations
import torch
from ..api import Generator, GeneratorInfo, ParamSpec
from ..utils import grid

class Gabor(Generator):
    @property
    def info(self) -> GeneratorInfo:
        return GeneratorInfo(
            name="GABOR",
            param_specs=(
                ParamSpec("freq", "float", (0.5, 64.0), "cycles/img", 0.5),
                ParamSpec("angle_deg", "float", (0.0, 180.0), "deg", 5.0),
                ParamSpec("sigma", "float", (0.1, 1.0), None, 0.05),
                ParamSpec("phase", "float", (0.0, 6.28318), "rad", 0.2),
            ),
            supports_noise=False,
        )

    @torch.no_grad()
    def render(self, tiles_hw, params, seeds, *, device, dtype):
        B = params.shape[0]
        h, w = tiles_hw
        xx, yy = grid(h, w, device=device, dtype=dtype)
        xx = xx.unsqueeze(0); yy = yy.unsqueeze(0)
        f   = params[:,0].view(B,1,1).to(dtype)
        ang = torch.deg2rad(params[:,1]).view(B,1,1).to(dtype)
        sig = params[:,2].view(B,1,1).to(dtype)
        ph  = params[:,3].view(B,1,1).to(dtype)
        xr =  xx*torch.cos(ang) + yy*torch.sin(ang)
        yr = -xx*torch.sin(ang) + yy*torch.cos(ang)
        env = torch.exp(-0.5*((xr/sig)**2 + (yr/sig)**2))
        s = torch.sin(2*torch.pi*f*xr + ph) * env
        return s.clamp(-1,1).unsqueeze(1)

GEN = Gabor()
