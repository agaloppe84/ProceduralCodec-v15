from __future__ import annotations
import torch
from ..api import Generator, GeneratorInfo, ParamSpec
from ..utils import grid
from ..noise import _hash2, rand01
from pc15core.rng import to_int64_signed  # ← helper public

# u64 → int64 signé (même bits) pour éviter l'overflow lors de l'XOR
_GOLDEN64_I = to_int64_signed(0x9E3779B97F4A7C15)


class Starfield(Generator):
    @property
    def info(self) -> GeneratorInfo:
        return GeneratorInfo(
            name="STARFIELD",
            param_specs=(
                ParamSpec("cells", "int", (32, 512), "per img", 128.0),
                ParamSpec("density", "float", (0.05, 1.0), None, 0.1),
                ParamSpec("sigma", "float", (0.005, 0.1), None, 0.02),
            ),
            supports_noise=True,
        )

    @torch.no_grad()
    def render(self, tiles_hw, params, seeds, *, device, dtype):
        B = params.shape[0]
        h, w = tiles_hw
        xx, yy = grid(h, w, device=device, dtype=dtype)
        xx = xx.unsqueeze(0)
        yy = yy.unsqueeze(0)

        cells = params[:, 0].view(B, 1, 1).to(dtype)
        dens  = params[:, 1].view(B, 1, 1).to(dtype)
        sigma = params[:, 2].view(B, 1, 1).to(dtype)
        seed64 = seeds.view(B, 1, 1).to(torch.int64)

        fx = (xx + 1.0) * cells * 0.5
        fy = (yy + 1.0) * cells * 0.5
        xi = torch.floor(fx).to(torch.int64)
        yi = torch.floor(fy).to(torch.int64)
        xf = fx - xi.to(dtype)
        yf = fy - yi.to(dtype)

        out = torch.full((B, h, w), -1.0, device=device, dtype=dtype)

        for oy in (-1, 0, 1):
            for ox in (-1, 0, 1):
                cx = xi + ox
                cy = yi + oy
                h2 = _hash2(cx, cy, seed64)

                keep = rand01(h2) < dens
                jx = rand01(h2) - 0.5
                jy = rand01(h2 ^ _GOLDEN64_I) - 0.5  # ← plus de torch.tensor(...)

                dx = (cx.to(dtype) + jx) - fx
                dy = (cy.to(dtype) + jy) - fy
                d2 = dx * dx + dy * dy

                star = torch.exp(-d2 / (2.0 * (sigma ** 2)))
                out = torch.maximum(out, torch.where(keep, star, torch.full_like(star, -1.0)))

        return out.unsqueeze(1).clamp(-1, 1)


GEN = Starfield()
