from __future____ import annotations
import torch
from ..api import Generator, GeneratorInfo, ParamSpec
from ..utils import grid

class WavesDir(Generator):
    @property
    def info(self) -> GeneratorInfo:
        return GeneratorInfo(
            name="WAVES_DIR",
            param_specs=(
                ParamSpec("freq", "float", (0.5, 64.0), "cycles/img", 0.5),
                ParamSpec("angle_deg", "float", (0.0, 180.0), "deg", 5.0),
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
        ang=torch.deg2rad(params[:,1]).view(B,1,1).to(dtype)
        amp=params[:,2].view(B,1,1).to(dtype)
        ph=params[:,3].view(B,1,1).to(dtype)
        u=xx*torch.cos(ang)+yy*torch.sin(ang)
        s=torch.sin(2*torch.pi*f*u + ph)*amp
        return s.clamp(-1,1).unsqueeze(1)

GEN = WavesDir()
