from __future__ import annotations
import torch
from ..api import Generator, GeneratorInfo, ParamSpec
from ..utils import grid

class Checker(Generator):
    @property
    def info(self) -> GeneratorInfo:
        return GeneratorInfo(
            name="CHECKER",
            param_specs=(
                ParamSpec("cells", "int", (2, 512), "per img", 2.0),
                ParamSpec("angle_deg", "float", (0.0, 180.0), "deg", 5.0),
            ),
            supports_noise=False,
        )

    @torch.no_grad()
    def render(self, tiles_hw, params, seeds, *, device, dtype):
        B = params.shape[0]
        h, w = tiles_hw
        xx, yy = grid(h, w, device=device, dtype=dtype)
        xx = xx.unsqueeze(0); yy = yy.unsqueeze(0)
        ang = torch.deg2rad(params[:,1]).view(B,1,1).to(dtype)
        c = torch.cos(ang); s = torch.sin(ang)
        xr = xx*c - yy*s
        yr = xx*s + yy*c
        cells = params[:,0].view(B,1,1).to(dtype)
        fx = (xr+1.0)*cells*0.5
        fy = (yr+1.0)*cells*0.5
        xi = torch.floor(fx).to(torch.int64)
        yi = torch.floor(fy).to(torch.int64)
        v = ((xi + yi) & 1).to(dtype)
        out = v*2.0 - 1.0
        return out.unsqueeze(1)

GEN = Checker()
