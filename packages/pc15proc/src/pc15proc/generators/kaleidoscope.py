import math
import torch
from ..api import Generator, GeneratorInfo, ParamSpec
from ..utils import grid

class Kaleidoscope(Generator):
    @property
    def info(self) -> GeneratorInfo:
        return GeneratorInfo(
            name="KALEIDOSCOPE",
            param_specs=(
                ParamSpec("sectors", "int", (3, 64), None, 12.0),
                ParamSpec("freq", "float", (4.0, 128.0), None, 32.0),
            ),
            supports_noise=False,
        )

    @torch.no_grad()
    def render(self, tiles_hw, params, seeds, *, device, dtype):
        B = params.shape[0]
        h, w = tiles_hw

        xx, yy = grid(h, w, device=device, dtype=dtype)  # (H,W)
        # -> batch
        xx = xx.unsqueeze(0).expand(B, -1, -1)           # (B,H,W)
        yy = yy.unsqueeze(0).expand(B, -1, -1)           # (B,H,W)

        sectors = params[:, 0].view(B, 1, 1).to(dtype)   # (B,1,1)
        freq    = params[:, 1].view(B, 1, 1).to(dtype)   # (B,1,1)

        theta = torch.atan2(yy, xx)                      # (B,H,W)
        sector_angle = (2.0 * math.pi) / sectors         # (B,1,1)

        # angle plié dans un secteur (miroir)
        t = torch.remainder(theta, sector_angle)         # (B,H,W)
        t = torch.minimum(t, sector_angle - t)           # (B,H,W)

        # un motif simple périodique (peu importe tant que [-1,1], déterministe)
        pat = torch.cos(freq * t)                        # (B,H,W)
        out = pat.clamp(-1, 1).unsqueeze(1)              # (B,1,H,W)
        return out

GEN = Kaleidoscope()
