from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Protocol
import torch

ParamDict = dict[str, Any]

@dataclass(frozen=True)
class ParamSpec:
    name: str
    type: str
    range: tuple[float, float] | None = None
    enum: tuple[Any, ...] | None = None
    units: str | None = None
    quant: float | None = None

@dataclass(frozen=True)
class GeneratorInfo:
    name: str
    param_specs: tuple[ParamSpec, ...]
    supports_noise: bool
    deterministic: bool = True

class Generator(Protocol):
    @property
    def info(self) -> GeneratorInfo: ...
    def render(
        self,
        tiles_hw: tuple[int, int],
        params: torch.Tensor,
        seeds: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor: ...

def render_bn(
    gen: Generator,
    tile_batch: torch.Tensor,
    params: torch.Tensor,
    seeds: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    N, _, h, w = tile_batch.shape
    outs = []
    for _ in range(N):
        outs.append(gen.render((h, w), params, seeds, device=device, dtype=dtype))
    return torch.stack(outs, dim=1)
