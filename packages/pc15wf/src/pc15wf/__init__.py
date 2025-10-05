# packages/pc15wf/src/pc15wf/__init__.py
from __future__ import annotations

from .api import atomic_write, pc15_name, log_append, detect_gpu

__all__ = [
    "atomic_write",
    "pc15_name",
    "log_append",
    "detect_gpu",
    # on n’importe PAS le sous-module cli ici pour éviter les imports lourds au top-level
]

__version__ = "15.0.0"
