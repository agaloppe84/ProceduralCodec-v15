from __future__ import annotations
from .api import Generator, GeneratorInfo
from pc15core.errors import UnknownGeneratorError

_REG: dict[str, Generator] = {}

def register(gen: Generator) -> None:
    _REG[gen.info.name] = gen

def get(name: str) -> Generator:
    try:
        return _REG[name]
    except KeyError as exc:
        raise UnknownGeneratorError(f"Générateur inconnu: {name}") from exc

def list_generators() -> list[GeneratorInfo]:
    return [g.info for g in _REG.values()]
