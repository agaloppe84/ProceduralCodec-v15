# packages/pc15codec/src/pc15codec/__init__.py
from __future__ import annotations

"""PC15 - codec v15 (public surface).

Expose uniquement la config publique et quelques constantes stables,
sans importer les modules lourds par défaut.
"""

__version__ = "15.0.0"

# API publique (stable)
from .config import CodecConfig
from .payload import ANS0_FMT, RAW_FMT
from .rans import DEFAULT_TABLE_ID

__all__ = [
    "__version__",
    "CodecConfig",
    "ANS0_FMT", "RAW_FMT",
    "DEFAULT_TABLE_ID",
]
