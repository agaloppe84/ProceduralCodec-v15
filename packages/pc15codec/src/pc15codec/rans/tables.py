from __future__ import annotations
from importlib import resources
from pathlib import Path
from typing import Any
from ..paths import PathsConfig

_PKG_NS = "pc15codec.data.rans"

def _load_json_file(p: Path) -> dict[str, Any]:
    import json
    with p.open("rb") as f:
        return json.load(f)

def load_table_by_id(table_id: str, paths: PathsConfig | None = None) -> dict[str, Any]:
    """Load a frozen rANS table by ID.

    Resolution order: explicit paths.models_dir → ENV → packaged resource.
    Packaged tables are read-only; learned tables live under models_dir.  # [ML]
    """
    # 1) explicit / ENV
    root = (paths.models() if paths else None) or PathsConfig.from_env().models()
    if root:
        p = Path(root) / "rans" / f"{table_id}.json"
        if p.is_file():
            return _load_json_file(p)
    # 2) packaged resource
    with resources.files(_PKG_NS).joinpath(f"{table_id}.json").open("rb") as f:
        import json
        return json.load(f)
