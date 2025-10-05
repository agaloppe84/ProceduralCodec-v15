from __future__ import annotations
import torch
from ..api import Generator, GeneratorInfo, ParamSpec
from ..utils import grid

class Kaleidoscope(Generator):
    @property
    def info(self) -> GeneratorInfo:
        return GeneratorInfo(
            name="KALEIDOSCOPE",
            param_specs=(
                ParamSpec("sectors", "int", (2, 24), None, 1.0),
                ParamSpec("freq", "float", (1.0, 64.0), "cycles/img", 0.5),
            ),
            supports_noise=False,
        )
    @torch.no_grad()
    def render(self, tiles_hw, params, seeds, *, device, dtype):
        B = params.shape[0]; h,w = tiles_hw
        xx,yy = grid(h,w,device=device,dtype=dtype); xx=xx.unsqueeze(0); yy=yy.unsqueeze(0)
        sectors = torch.clamp(params[:,0], 2, 24).to(torch.int64).view(B,1,1)
        f = params[:,1].view(B,1,1).to(dtype)
        theta = torch.atan2(yy, xx)  # [-pi, pi]
        r = torch.sqrt(xx*xx + yy*yy)
        out = torch.zeros((B,h,w), device=device, dtype=dtype)
        for i in range(B):
            s = sectors[i,0,0].item()
            sector_angle = (2*torch.pi) / float(s)
            t = (theta[i] % sector_angle)
            t = torch.minimum(t, sector_angle - t)  # mirror
            u = torch.cos(t) * r[i]
            out[i] = torch.sin(2*torch.pi*f[i]*u)
        return out.clamp(-1,1).unsqueeze(1)

GEN = Kaleidoscope()
