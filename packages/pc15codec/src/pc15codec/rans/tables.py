# packages/pc15codec/src/pc15codec/rans/tables.py
from __future__ import annotations

import os
from importlib import resources
from pathlib import Path
from typing import Any, Optional, List, Tuple

from ..paths import PathsConfig

__all__ = [
    "DEFAULT_TABLE_ID",
    "load_table_by_id",
    "available_tables",
]

# Espace-ressource packagé (tables gelées incluses dans la wheel)
_PKG_NS = "pc15codec.data.rans"

# Table par défaut pour v15
DEFAULT_TABLE_ID = "v15_default"

# Petit cache (clé = (env_root, models_root, table_id))
_CACHE: dict[Tuple[str, str, str], dict[str, Any]] = {}


def _env_custom_root() -> Optional[Path]:
    """
    Retourne la racine explicite des tables si l'ENV `PC15_RANS_TABLES` est défini,
    sinon None.

    Priorité : si défini, ce dossier est **toujours** utilisé en premier.
    """
    v = os.getenv("PC15_RANS_TABLES")
    return Path(v) if v else None


def _load_json_file(p: Path) -> dict[str, Any]:
    """
    Charge un fichier JSON (binaire) et retourne le dict Python correspondant.
    Hypothèse : le fichier est bien formé (ValueError sinon).
    """
    import json
    with p.open("rb") as f:
        return json.load(f)


def load_table_by_id(table_id: str, paths: Optional[PathsConfig] = None) -> dict[str, Any]:
    """
    Charge une table rANS gelée par son identifiant.

    Ordre de résolution (du plus prioritaire au moins prioritaire) :
      0) ENV `PC15_RANS_TABLES` → <root>/<table_id>.json
      1) Dossier models (paths.models() ou ENV gérée par PathsConfig)
         → <models_root>/rans/<table_id>.json
      2) Ressource packagée (read-only) sous `pc15codec/data/rans/<table_id>.json`

    Caching :
      - Résultat mis en cache mémoire selon la clé (env_root, models_root, table_id)
        pour éviter des I/O répétées.
      - Changer d’ENV ou de models dir invalide implicitement la clé et donc contourne le cache.

    Paramètres
    ----------
    table_id : str
        Identifiant logique de la table (ex: "v15_default").
    paths : PathsConfig | None
        Configuration explicite des chemins. Si None, on dérive depuis l’ENV via PathsConfig.

    Retour
    ------
    dict[str, Any]
        Dictionnaire contenant au minimum les champs :
        - "precision": int (1..15)
        - "freqs": List[int] de taille 256, somme = 1<<precision
        - "cdf": List[int] de taille 257, cdf[256] == somme(freqs)

    Exceptions
    ----------
    FileNotFoundError si aucune source ne fournit `table_id`.
    ValueError si le contenu JSON est invalide.
    """
    # Racine 0) ENV PC15_RANS_TABLES
    env_root = _env_custom_root()

    # Racine 1) models dir (explicite ou dérivée de l'ENV via PathsConfig)
    models_root = (paths.models() if paths else None) or PathsConfig.from_env().models()

    # Clé cache (stringifiée pour être hashable et stable)
    key = (str(env_root) if env_root else "", str(models_root) if models_root else "", table_id)
    if key in _CACHE:
        return _CACHE[key]

    # 0) ENV PC15_RANS_TABLES
    if env_root:
        p = Path(env_root) / f"{table_id}.json"
        if p.is_file():
            tbl = _load_json_file(p)
            _CACHE[key] = tbl
            return tbl

    # 1) models dir
    if models_root:
        p = Path(models_root) / "rans" / f"{table_id}.json"
        if p.is_file():
            tbl = _load_json_file(p)
            _CACHE[key] = tbl
            return tbl

    # 2) Ressource packagée
    try:
        with resources.files(_PKG_NS).joinpath(f"{table_id}.json").open("rb") as f:
            import json
            tbl = json.load(f)
            _CACHE[key] = tbl
            return tbl
    except FileNotFoundError:
        pass  # homogénéiser l'erreur à la fin

    raise FileNotFoundError(
        f"rANS table '{table_id}' not found in "
        f"PC15_RANS_TABLES={env_root!s} nor models_root={models_root!s} nor package '{_PKG_NS}'."
    )


def _list_json_stems(root: Path) -> List[str]:
    """
    Retourne la liste des noms de fichiers *.json **sans** l'extension dans `root`.
    Si le dossier n'existe pas, retourne [].
    """
    if not root or not root.exists():
        return []
    return sorted({p.stem for p in root.glob("*.json") if p.is_file()})


def available_tables(paths: Optional[PathsConfig] = None) -> List[str]:
    """
    Énumère les tables disponibles en agrégeant les trois sources :
      - ENV `PC15_RANS_TABLES` (si défini)      — priorité 0
      - <models_root>/rans (si disponible)       — priorité 1
      - ressources packagées `pc15codec.data.rans` — priorité 2

    L’ordre de la liste retournée est alphabétique et **dédupliqué**.

    Paramètres
    ----------
    paths : PathsConfig | None
        Configuration explicite des chemins pour models_root.

    Retour
    ------
    list[str]
        Liste d’identifiants (stems) disponibles, ex. ["v15_default", "custom_A", ...].
    """
    out: set[str] = set()

    # 0) ENV
    env_root = _env_custom_root()
    if env_root:
        out.update(_list_json_stems(env_root))

    # 1) models dir
    models_root = (paths.models() if paths else None) or PathsConfig.from_env().models()
    if models_root:
        out.update(_list_json_stems(Path(models_root) / "rans"))

    # 2) package
    try:
        pkg_root = resources.files(_PKG_NS)
        for entry in pkg_root.iterdir():
            # Certaines implémentations renvoient des Traversable (pas Path)
            if entry.name.endswith(".json"):
                out.add(Path(entry.name).stem)
    except FileNotFoundError:
        # Paquet sans ressources (dev editable) — ok, on ignore
        pass

    return sorted(out)
