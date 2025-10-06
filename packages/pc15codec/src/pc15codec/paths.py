from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os

@dataclass(frozen=True)
class PathsConfig:
    """Parametric path resolver for models, outputs, artifacts, datasets.

    ENV keys
    --------
    PC15_MODELS_DIR    → tables rANS, QV codebooks
    PC15_OUTPUTS_DIR   → bitstreams/reconstructions           # [STORE:OVERWRITE]
    PC15_ARTIFACTS_DIR → figures RD, CSV, logs (append-only)  # [STORE:CUMULATIVE]
    PC15_DATASETS_DIR  → external datasets (read-only)
    """
    models_dir: Path | None = None
    outputs_dir: Path | None = None   # [STORE:OVERWRITE]
    artifacts_dir: Path | None = None # [STORE:CUMULATIVE]
    datasets_dir: Path | None = None

    @staticmethod
    def from_env() -> "PathsConfig":
        return PathsConfig(
            models_dir=_opt_env("PC15_MODELS_DIR"),
            outputs_dir=_opt_env("PC15_OUTPUTS_DIR"),
            artifacts_dir=_opt_env("PC15_ARTIFACTS_DIR"),
            datasets_dir=_opt_env("PC15_DATASETS_DIR"),
        )

    # Accessors (explicit → ENV fallback)
    def models(self) -> Path | None: return self.models_dir or _opt_env("PC15_MODELS_DIR")
    def outputs(self) -> Path | None: return self.outputs_dir or _opt_env("PC15_OUTPUTS_DIR")
    def artifacts(self) -> Path | None: return self.artifacts_dir or _opt_env("PC15_ARTIFACTS_DIR")
    def datasets(self) -> Path | None: return self.datasets_dir or _opt_env("PC15_DATASETS_DIR")

    # Builders
    def outputs_path(self, *parts: str, create: bool = False) -> Path | None:
        root = self.outputs()
        if not root: return None
        p = root / Path(*parts)
        if create: p.parent.mkdir(parents=True, exist_ok=True)
        return p  # [STORE:OVERWRITE]

    def artifacts_path(self, *parts: str, create: bool = False) -> Path | None:
        root = self.artifacts()
        if not root: return None
        p = root / Path(*parts)
        if create: p.parent.mkdir(parents=True, exist_ok=True)
        return p  # [STORE:CUMULATIVE]

def _opt_env(name: str) -> Path | None:
    v = os.getenv(name)
    return Path(v) if v else None

__all__ = ["PathsConfig"]
