from __future__ import annotations
import math
import torch
from .api import Generator, GeneratorInfo, ParamSpec
from .utils import grid

class Stripes(Generator):
    @property
    def info(self) -> GeneratorInfo:
        return GeneratorInfo(
            name="STRIPES",
            param_specs=(
                ParamSpec("freq", "float", (0.5, 64.0), units="cycles/img", quant=0.5),
                ParamSpec("angle_deg", "float", (0.0, 180.0), units="deg", quant=15.0),
                ParamSpec("phase", "float", (0.0, 6.28318), units="rad", quant=1.5708),
            ),
            supports_noise=False,
        )

    def render(self, tiles_hw: tuple[int, int], params: torch.Tensor, seeds: torch.Tensor, *, device, dtype):
        B = params.shape[0]
        h, w = tiles_hw
        xx, yy = grid(h, w, device=device, dtype=dtype)
        out = torch.empty((B, 1, h, w), device=device, dtype=dtype)
        for i in range(B):
            freq = float(params[i, 0].item())
            ang = math.radians(float(params[i, 1].item()))
            phase = float(params[i, 2].item())
            u = xx * math.cos(ang) + yy * math.sin(ang)
            s = torch.sin(2 * math.pi * freq * u + phase)
            out[i, 0] = s.clamp(-1, 1)
        return out

GEN = Stripes()
