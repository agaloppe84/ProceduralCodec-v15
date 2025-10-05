from __future__ import annotations

from dataclasses import dataclass
import torch

from .api import GeneratorInfo
from pc15core.errors import InvalidParamsError


@dataclass
class ParamCodec:
    info: GeneratorInfo

    def _specs(self) -> dict[str, any]:
        return {p.name: p for p in self.info.param_specs}

    # -------------------------
    # Validation "lib propre"
    # -------------------------
    def validate(self, params: dict[str, any]) -> None:
        specs = self._specs()
        for k, v in params.items():
            if k not in specs:
                raise InvalidParamsError(f"Unknown param '{k}' for {self.info.name}")
            p = specs[k]
            if p.type in ("float", "int"):
                if p.range is not None:
                    lo, hi = p.range
                    x = float(v)
                    if not (float(lo) <= x <= float(hi)):
                        raise InvalidParamsError(f"{k}={x} ∉ [{lo}, {hi}]")
            elif p.type == "enum":
                if p.enum is None:
                    raise InvalidParamsError(f"{k} is enum but has no choices")
                if v not in p.enum:
                    raise InvalidParamsError(f"{k}={v!r} not in {p.enum}")
            elif p.type == "bool":
                _ = bool(v)  # just check castable
            else:
                raise InvalidParamsError(f"Unsupported param type '{p.type}' for {k}")

        # also ensure all required keys provided (strict)
        for p in self.info.param_specs:
            if p.name not in params:
                raise InvalidParamsError(f"Missing required param '{p.name}'")

    # -------------------------
    # Encodage tensorisé
    # -------------------------
    def to_tensor(self, params: dict[str, any], device, dtype) -> torch.Tensor:
        """Encode dict params -> vecteur tensorisé (float) suivant l'ordre de param_specs.
           Conventions:
             - float/int -> valeur numérique
             - bool -> 1.0 / 0.0
             - enum -> index (float) dans p.enum
        """
        self.validate(params)
        vec: list[float] = []
        for p in self.info.param_specs:
            v = params[p.name]
            if p.type == "bool":
                vec.append(1.0 if bool(v) else 0.0)
            elif p.type == "enum":
                assert p.enum is not None
                try:
                    idx = p.enum.index(v)
                except ValueError:
                    raise InvalidParamsError(f"{p.name}={v!r} not in {p.enum}")
                vec.append(float(idx))
            elif p.type == "int":
                vec.append(float(int(v)))
            else:  # "float"
                vec.append(float(v))
        return torch.tensor(vec, device=device, dtype=dtype)

    # -------------------------
    # Décodage tensor -> dict
    # -------------------------
    def from_tensor(self, vec: torch.Tensor) -> dict[str, any]:
        """Decode vecteur -> dict params (int arrondi, bool seuil 0.5, enum par index)."""
        out: dict[str, any] = {}
        for i, p in enumerate(self.info.param_specs):
            x = float(vec[i].item())
            if p.type == "int":
                out[p.name] = int(round(x))
            elif p.type == "bool":
                out[p.name] = bool(x >= 0.5)
            elif p.type == "enum":
                assert p.enum is not None
                idx = int(round(x))
                if idx < 0:
                    idx = 0
                if idx >= len(p.enum):
                    idx = len(p.enum) - 1
                out[p.name] = p.enum[idx]
            else:  # float
                out[p.name] = x
        return out

    # -------------------------
    # Petite grille d’exploration (optionnel)
    # -------------------------
    def grid(self) -> list[dict[str, any]]:
        """Construit une grille coarse (2 valeurs / float & int, 1ère valeur pour enum/bool).
        Utile pour du smoke test ou un coarse search minimal.
        """
        gs: list[dict[str, any]] = []
        # point milieu
        mid: dict[str, any] = {}
        for p in self.info.param_specs:
            if p.type in ("float", "int") and p.range is not None:
                lo, hi = p.range
                x = (float(lo) + float(hi)) / 2.0
                if p.type == "int":
                    x = int(round(x))
                mid[p.name] = x
            elif p.type == "enum" and p.enum is not None:
                mid[p.name] = p.enum[0]
            elif p.type == "bool":
                mid[p.name] = False
            else:
                mid[p.name] = 0.0
        gs.append(mid)

        # variante aux bords pour float/int
        edge: dict[str, any] = {}
        for p in self.info.param_specs:
            if p.type in ("float", "int") and p.range is not None:
                lo, hi = p.range
                edge[p.name] = lo if isinstance(lo, (int, float)) else mid[p.name]
            else:
                edge[p.name] = mid[p.name]
        gs.append(edge)
        return gs
