from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch
from .api import GeneratorInfo
from pc15core.errors import InvalidParamsError

@dataclass
class ParamCodec:
    info: GeneratorInfo

    def validate(self, params: dict[str, Any]) -> None:
        specs = {p.name: p for p in self.info.param_specs}
        for k in params:
            if k not in specs:
                raise InvalidParamsError(f"Paramètre inattendu: {k}")
        for p in self.info.param_specs:
            if p.name not in params:
                raise InvalidParamsError(f"Paramètre manquant: {p.name}")
            val = params[p.name]
            if p.type in ("float", "int") and p.range is not None:
                lo, hi = p.range
                if not (lo <= float(val) <= hi):
                    raise InvalidParamsError(f"{p.name} hors bornes [{lo},{hi}]")
            if p.type == "enum" and p.enum is not None and val not in p.enum:
                raise InvalidParamsError(f"{p.name} ∉ {p.enum}")

    def to_tensor(self, params: dict[str, Any], device, dtype) -> torch.Tensor:
        self.validate(params)
        vec: list[float] = []
        for p in self.info.param_specs:
            v = params[p.name]
            if p.type == "bool":
                v = 1.0 if bool(v) else 0.0
            else:
                v = float(v)
            vec.append(v)
        return torch.tensor(vec, device=device, dtype=dtype)

    def from_tensor(self, vec: torch.Tensor) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for i, p in enumerate(self.info.param_specs):
            x = float(vec[i].item())
            if p.type == "int":
                x = int(round(x))
            if p.type == "bool":
                x = bool(x >= 0.5)
            out[p.name] = x
        return out

    def grid(self, coarse: bool = True) -> torch.Tensor:
        device = torch.device("cpu")
        dtype = torch.float32
        axes = []
        for p in self.info.param_specs:
            if p.type in ("float", "int") and p.range is not None and p.quant:
                lo, hi = p.range
                step = max(p.quant, 1e-6)
                n = int(max(1, round((hi - lo) / step))) + 1
                axes.append(torch.linspace(lo, hi, n, device=device, dtype=dtype))
            elif p.type == "enum" and p.enum is not None:
                axes.append(torch.tensor([float(v) for v in p.enum], device=device, dtype=dtype))
            else:
                axes.append(torch.tensor([0.0], device=device, dtype=dtype))
        mesh = torch.meshgrid(*axes, indexing="ij")
        grid = torch.stack([m.reshape(-1) for m in mesh], dim=1)
        return grid
