from __future__ import annotations

import importlib
import pkgutil
from typing import Iterable

from .registry import register

def register_from_package(pkg_name: str = "pc15proc.generators",
                          expect_var: str = "GEN",
                          verbose: bool = False) -> list[str]:
    """
    Importe tous les modules d’un package et enregistre tout objet `GEN`
    (instance conforme à l’interface Generator).
    Retourne la liste des noms effectivement enregistrés.
    """
    pkg = importlib.import_module(pkg_name)
    registered: list[str] = []

    for mod in pkgutil.iter_modules(pkg.__path__):  # type: ignore[attr-defined]
        if mod.ispkg or mod.name.startswith("_"):
            continue
        module = importlib.import_module(f"{pkg_name}.{mod.name}")
        if hasattr(module, expect_var):
            gen = getattr(module, expect_var)
            # Duck-typing doux : doit avoir .info.name et .render
            if hasattr(gen, "info") and hasattr(gen.info, "name") and hasattr(gen, "render"):
                register(gen)
                registered.append(gen.info.name)
                if verbose:
                    print(f"[pc15proc] registered {gen.info.name} from {module.__name__}")
            else:
                if verbose:
                    print(f"[pc15proc] skip {module.__name__} (GEN missing .info/.render)")
        else:
            if verbose:
                print(f"[pc15proc] no GEN in {module.__name__}")
    return sorted(registered)
