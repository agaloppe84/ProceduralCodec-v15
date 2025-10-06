from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import os, platform
from typing import Dict

@dataclass
class PathsConfig:
    base: str
    dataset_images: str
    artifacts_subdir: str = "artifacts"
    cache_subdir: str = ".cache"
    labels_dir: str | None = None

    @property
    def base_dir(self) -> Path: return Path(self.base)
    @property
    def artifacts_dir(self) -> Path: return self.base_dir / self.artifacts_subdir
    @property
    def cache_dir(self) -> Path: return self.base_dir / self.cache_subdir

    def ensure(self) -> None:
        for p in (self.base_dir, self.artifacts_dir, self.cache_dir, Path(self.dataset_images)):
            p.mkdir(parents=True, exist_ok=True)
        if self.labels_dir: Path(self.labels_dir).mkdir(parents=True, exist_ok=True)

    def dict(self) -> Dict:
        d = asdict(self)
        d["artifacts_dir"] = str(self.artifacts_dir)
        d["cache_dir"] = str(self.cache_dir)
        return d

def env_summary() -> Dict:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
    }
    for k in ["PC15_BASE_DIR","PC15_DATA_DIR","PC15_OUT_DIR","PC15_CACHE_DIR",
              "PC15_DATASET_IMAGES","PC15_LABELS_DIR"]:
        if k in os.environ: info[k] = os.environ[k]
    return info
