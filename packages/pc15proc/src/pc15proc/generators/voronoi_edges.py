from __future__ import annotations
import torch
from ..api import Generator, GeneratorInfo, ParamSpec
from ..utils import grid
from ..noise import _hash2, rand01
from pc15core.rng import to_int64_signed  # ← helper public (u64 -> i64 signé)

# remap u64 -> int64 signé (mêmes bits), réutilisé pour l’XOR
_GOLDEN64_I = to_int64_signed(0x9E3779B97F4A7C15)


class VoronoiEdges(Generator):
    @property
    def info(self) -> GeneratorInfo:
        return GeneratorInfo(
            name="VORONOI_EDGES",
            param_specs=(
                ParamSpec("scale", "int", (16, 256), "per img", 64.0),
                ParamSpec("sharp", "float", (0.5, 10.0), None, 2.0),
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

        sc = params[:, 0].view(B, 1, 1).to(dtype)
        sharp = params[:, 1].view(B, 1, 1).to(dtype)
        seed64 = seeds.view(B, 1, 1).to(torch.int64)

        fx = (xx + 1.0) * (sc / 2.0)
        fy = (yy + 1.0) * (sc / 2.0)
        xi = torch.floor(fx).to(torch.int64)
        yi = torch.floor(fy).to(torch.int64)
        xf = fx - xi.to(dtype)
        yf = fy - yi.to(dtype)

        ds = []
        for oy in (-1, 0, 1):
            for ox in (-1, 0, 1):
                cx = xi + ox
                cy = yi + oy
                h2 = _hash2(cx, cy, seed64)
                # jitter pseudo-aléatoire dans la même cellule
                jx = rand01(h2)
                jy = rand01(h2 ^ _GOLDEN64_I)  # ← plus de torch.tensor(...)

                dx = (cx.to(dtype) + jx) - fx
                dy = (cy.to(dtype) + jy) - fy
                d2 = dx * dx + dy * dy
                ds.append(d2)

        # distance aux 2 sites les plus proches → “edges” (Voronoi)
        # on empile puis on prend les deux plus petites distances
        D = torch.stack(ds, dim=0)  # [9, B, h, w]
        d_sorted, _ = torch.sort(D, dim=0)
        d1 = d_sorted[0]
        d2 = d_sorted[1]
        edges = (d2 - d1)  # contraste sur l’écart des deux plus proches
        out = (1.0 - torch.exp(-sharp * edges)).unsqueeze(1) * 2.0 - 1.0
        return out.clamp(-1, 1)


GEN = VoronoiEdges()
