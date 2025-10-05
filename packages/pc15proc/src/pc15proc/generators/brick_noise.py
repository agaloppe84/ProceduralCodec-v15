from __future__ import annotations
import torch
from ..api import Generator, GeneratorInfo, ParamSpec
from ..utils import grid
from ..noise import _hash2, rand01

class BrickNoise(Generator):
    @property
    def info(self) -> GeneratorInfo:
        return GeneratorInfo(
            name="BRICK_NOISE",
            param_specs=(
                ParamSpec("rows", "int", (2, 512), "per img", 2.0),
                ParamSpec("cols", "int", (2, 512), "per img", 2.0),
                ParamSpec("mortar", "float", (0.002, 0.1), None, 0.002),
                ParamSpec("contrast", "float", (0.5, 2.0), None, 0.1),
            ),
            supports_noise=True,
        )
    @torch.no_grad()
    def render(self, tiles_hw, params, seeds, *, device, dtype):
        B=params.shape[0]; h,w=tiles_hw
        xx,yy = grid(h,w,device=device,dtype=dtype); xx=xx.unsqueeze(0); yy=yy.unsqueeze(0)
        rows=params[:,0].view(B,1,1).to(dtype); cols=params[:,1].view(B,1,1).to(dtype)
        mortar=params[:,2].view(B,1,1).to(dtype); c=params[:,3].view(B,1,1).to(dtype)
        seed64=seeds.view(B,1,1).to(torch.int64)
        fy=(yy+1.0)*rows*0.5; fx=(xx+1.0)*cols*0.5
        yi=torch.floor(fy).to(torch.int64); xi=torch.floor(fx).to(torch.int64)
        fx_shift=fx + ((yi & 1).to(dtype) * 0.5)
        xi2=torch.floor(fx_shift).to(torch.int64)
        dx=torch.abs(fx_shift - xi2.to(dtype) - 0.5)
        dy=torch.abs(fy - yi.to(dtype) - 0.5)
        mortar_mask = ((dx <= mortar) | (dy <= mortar)).to(dtype)

        # per-brick hash → brightness
        hcell = _hash2(xi2, yi, seed64)
        val = rand01(hcell)  # in [0,1)
        val = ( (val - 0.5) * c + 0.5 ).clamp(0,1)
        out = val*2.0 - 1.0
        # impose mortar to -1.0
        out = torch.where(mortar_mask>0, torch.tensor(-1.0, device=device, dtype=dtype), out)
        return out.unsqueeze(1)

GEN = BrickNoise()
