from __future__ import annotations
from .discovery import register_from_package

def register_all(verbose: bool = False) -> list[str]:
    """Enregistre tous les générateurs trouvés dans pc15proc.generators."""
    return register_from_package("pc15proc.generators", verbose=verbose)
