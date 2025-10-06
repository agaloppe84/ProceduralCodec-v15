
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os

__all__ = ["PathsConfig", "resolve_path", "detect_default_base", "env_summary"]

def resolve_path(p: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(p))).resolve()

def detect_default_base() -> str:
    drive = Path("/content/drive/MyDrive/pc15")
    if drive.exists():
        return str(drive)
    colab = Path("/content")
    if colab.exists():
        return "/content"
    return "."

@dataclass
class PathsConfig:
    base_dir: str | None = None
    data_dir: str | None = None
    out_dir:  str | None = None
    cache_dir:str | None = None

    def __post_init__(self):
        base = self.base_dir or os.environ.get("PC15_BASE_DIR") or detect_default_base()
        data = self.data_dir or os.environ.get("PC15_DATA_DIR") or (str(Path(base) / "datasets"))
        out  = self.out_dir  or os.environ.get("PC15_OUT_DIR")  or (str(Path(base) / "artifacts"))
        cache= self.cache_dir or os.environ.get("PC15_CACHE_DIR") or (str(Path(base) / ".cache"))
        self.base_dir, self.data_dir, self.out_dir, self.cache_dir = base, data, out, cache

    @property
    def base(self) -> Path:  return resolve_path(self.base_dir)
    @property
    def data(self) -> Path:  return resolve_path(self.data_dir)
    @property
    def out(self)  -> Path:  return resolve_path(self.out_dir)
    @property
    def cache(self)-> Path:  return resolve_path(self.cache_dir)

    def ensure_dirs(self):
        self.data.mkdir(parents=True, exist_ok=True)
        self.out.mkdir(parents=True, exist_ok=True)
        self.cache.mkdir(parents=True, exist_ok=True)
        return self

    def to_dict(self):
        return {"base": str(self.base), "data": str(self.data), "out": str(self.out), "cache": str(self.cache)}

def env_summary() -> dict:
    keys = ["PC15_BASE_DIR","PC15_DATA_DIR","PC15_OUT_DIR","PC15_CACHE_DIR"]
    return {k: os.environ.get(k) for k in keys}
