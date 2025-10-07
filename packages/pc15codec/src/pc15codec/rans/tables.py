# packages/pc15codec/src/pc15codec/rans/tables.py
from __future__ import annotations

import math
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


# ---------- Coercion vers le format canonique attendu par rans_impl ----------

def _normalize_counts_to_freqs(counts: List[int], P: int) -> List[int]:
    """
    Convertit un histogramme `counts` (taille 256) en fréquences entières `freqs`
    de somme `base = 1<<P`, en garantissant au moins 1 pour chaque symbole présent.
    Stratégie: “largest remainder” + garde-fous déterministes.
    """
    base = 1 << int(P)
    if len(counts) != 256:
        raise ValueError("counts must have length 256")
    total = int(sum(max(0, int(c)) for c in counts))
    if total <= 0:
        # Tout à zéro -> met toute la masse sur 0 pour conserver une table valide
        freqs = [0] * 256
        freqs[0] = base
        return freqs

    # Au moins 1 pour les symboles présents
    raw = [0] * 256
    remainders: List[Tuple[float, int]] = []
    s = 0
    for i, h in enumerate(counts):
        h = int(h)
        if h <= 0:
            continue
        f = (h * base) / total
        q = int(f)
        if q == 0:
            q = 1
        raw[i] = q
        s += q
        remainders.append((f - q, i))

    # Ajustements pour égaliser à base
    if s > base:
        # Retirer d'abord aux plus petites fractions (puis aux plus gros si besoin)
        remainders.sort(key=lambda x: (x[0], counts[x[1]]))  # petites fractions d'abord
        i = 0
        while s > base and i < len(remainders):
            _, sym = remainders[i]
            if raw[sym] > 1:
                raw[sym] -= 1
                s -= 1
            i += 1
        if s > base:
            order = sorted(range(256), key=lambda k: raw[k], reverse=True)
            j = 0
            while s > base and j < 256:
                k = order[j]
                if raw[k] > 1:
                    raw[k] -= 1
                    s -= 1
                j += 1

    if s < base:
        # Ajouter aux plus grandes fractions (puis aux plus fréquents)
        remainders.sort(key=lambda x: (-x[0], -counts[x[1]]))  # grandes fractions d'abord
        i = 0
        while s < base and i < len(remainders):
            _, sym = remainders[i]
            raw[sym] += 1
            s += 1
            i += 1
        if s < base:
            order = sorted(range(256), key=lambda k: counts[k], reverse=True)
            j = 0
            while s < base and j < 256:
                k = order[j]
                raw[k] += 1
                s += 1
                j += 1

    # garde-fou final
    delta = base - sum(raw)
    if delta != 0:
        raw[0] += delta
    return raw


def _build_cdf_from_freqs(freqs: List[int]) -> List[int]:
    """
    Construit une CDF de taille 257 (cdf[256] == base) à partir de `freqs` (taille 256).
    """
    if len(freqs) != 256:
        raise ValueError("freqs must have length 256")
    base = sum(int(f) for f in freqs)
    cdf = [0] * 257
    acc = 0
    for i in range(256):
        cdf[i] = acc
        acc += int(freqs[i])
    cdf[256] = acc
    if acc != base:
        raise ValueError("cdf sum mismatch")
    return cdf


def _choose_precision(total: int, n_nonzero: int) -> int:
    """
    Choisit une précision P dans [1..15] telle que base=2^P soit suffisante
    pour représenter au moins 1 par symbole présent et approx. proportionnelle à total.
    """
    if n_nonzero <= 0:
        return 1  # arbitraire mais valide
    target = max(int(total), int(n_nonzero))
    P = max(1, int(math.ceil(math.log2(max(1, target)))))
    return min(P, 15)


def _coerce_to_core_tables(tbl: dict[str, Any]) -> dict[str, Any]:
    """
    Convertit un dict JSON quelconque (ex: {counts: [...], alphabet: ...})
    vers le format attendu par rans_impl:
        {"precision": P, "freqs": [256 ints], "cdf": [257 ints]}
    """
    # 1) Déjà au format canonique ?
    if all(k in tbl for k in ("precision", "freqs", "cdf")):
        # Sanity minimal
        P = int(tbl["precision"])
        freqs = list(map(int, tbl["freqs"]))
        cdf = list(map(int, tbl["cdf"]))
        if len(freqs) != 256 or len(cdf) != 257:
            raise ValueError("invalid canonical table shapes")
        if sum(freqs) != (1 << P):
            raise ValueError("freqs must sum to (1<<precision)")
        if cdf[256] != sum(freqs):
            raise ValueError("cdf mismatch with freqs")
        return {"precision": P, "freqs": freqs, "cdf": cdf}

    # 2) Cas “uniforme” déclaré (sans counts)
    if tbl.get("alphabet") == "byte_256_uniform":
        P = 8  # base=256
        freqs = [1] * 256
        cdf = _build_cdf_from_freqs(freqs)
        return {"precision": P, "freqs": freqs, "cdf": cdf}

    # 3) Cas "counts" (ou "hist") → normalisation vers base=2^P
    counts = None
    if "counts" in tbl and isinstance(tbl["counts"], list):
        counts = list(map(int, tbl["counts"]))
    elif "hist" in tbl and isinstance(tbl["hist"], list):
        counts = list(map(int, tbl["hist"]))

    if counts is None:
        raise ValueError("unknown rANS table schema (expected canonical or counts/hist)")

    # Normalisation
    if len(counts) != 256:
        # Certains schémas peuvent donner 'alphabet_size'; ici on impose 256
        if int(tbl.get("alphabet_size", 256)) != 256:
            raise ValueError("only byte-alphabet (256) supported")
        # Pad/truncate si besoin (rare)
        counts = (counts + [0] * 256)[:256]

    total = sum(max(0, c) for c in counts)
    n_nonzero = sum(1 for c in counts if c > 0)
    P = _choose_precision(total, n_nonzero)
    freqs = _normalize_counts_to_freqs(counts, P)
    cdf = _build_cdf_from_freqs(freqs)
    return {"precision": P, "freqs": freqs, "cdf": cdf}


# ------------------------------- API publique --------------------------------

def load_table_by_id(table_id: str, paths: Optional[PathsConfig] = None) -> dict[str, Any]:
    """
    Charge une table rANS gelée par son identifiant et la **convertit**
    vers le format canonique attendu par rans_impl.

    Ordre de résolution (du plus prioritaire au moins prioritaire) :
      0) ENV `PC15_RANS_TABLES` → <root>/<table_id>.json
      1) Dossier models (paths.models() ou ENV gérée par PathsConfig)
         → <models_root>/rans/<table_id>.json
      2) Ressource packagée (read-only) sous `pc15codec/data/rans/<table_id>.json`

    Caching :
      - Résultat mis en cache mémoire selon la clé (env_root, models_root, table_id)
        pour éviter des I/O répétées (après coercition).
      - Changer d’ENV ou de models dir invalide implicitement la clé.

    Exceptions
    ----------
    FileNotFoundError si aucune source ne fournit `table_id`.
    ValueError si le contenu JSON est invalide ou non convertible.
    """
    env_root = _env_custom_root()
    models_root = (paths.models() if paths else None) or PathsConfig.from_env().models()

    key = (str(env_root) if env_root else "", str(models_root) if models_root else "", table_id)
    if key in _CACHE:
        return _CACHE[key]

    # 0) ENV
    if env_root:
        p = Path(env_root) / f"{table_id}.json"
        if p.is_file():
            raw = _load_json_file(p)
            coerced = _coerce_to_core_tables(raw)
            _CACHE[key] = coerced
            return coerced

    # 1) models dir
    if models_root:
        p = Path(models_root) / "rans" / f"{table_id}.json"
        if p.is_file():
            raw = _load_json_file(p)
            coerced = _coerce_to_core_tables(raw)
            _CACHE[key] = coerced
            return coerced

    # 2) Ressource packagée
    try:
        with resources.files(_PKG_NS).joinpath(f"{table_id}.json").open("rb") as f:
            import json
            raw = json.load(f)
            coerced = _coerce_to_core_tables(raw)
            _CACHE[key] = coerced
            return coerced
    except FileNotFoundError:
        pass

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

    Retour
    ------
    list[str]
        Liste d’identifiants (stems) disponibles, ex. ["v15_default", "custom_A", ...].
    """
    out: set[str] = set()

    env_root = _env_custom_root()
    if env_root:
        out.update(_list_json_stems(env_root))

    models_root = (paths.models() if paths else None) or PathsConfig.from_env().models()
    if models_root:
        out.update(_list_json_stems(Path(models_root) / "rans"))

    try:
        pkg_root = resources.files(_PKG_NS)
        for entry in pkg_root.iterdir():
            if entry.name.endswith(".json"):
                out.add(Path(entry.name).stem)
    except FileNotFoundError:
        pass

    return sorted(out)
