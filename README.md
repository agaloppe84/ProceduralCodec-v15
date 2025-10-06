[![CI](https://github.com/agaloppe84/ProceduralCodec-v15/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/agaloppe84/ProceduralCodec-v15/actions/workflows/ci.yml)

# ProceduralCodec v15 — Codec procédural par tuiles (PC15)

**PC15** est un codec d’image **entièrement procédural** : l’image est un **programme** (suite de générateurs + paramètres quantifiés + seeds).
L’encodage sélectionne, **tuile par tuile** (GPU), le meilleur couple *(générateur, paramètres, seed)* via un score **RD** (distorsion + coût en bits) et produit un bitstream **déterministe et idempotent**.

> ✅ **État v15.0.4** : encode/décode **Y**, bitstream **v15** (rANS auto-portant), tiling/blend, métriques, orchestration.
> 🧠 **Pré-intégration IA (v15.1+)** : hooks *optionnels* côté encodeur (sélecteur de générateurs / init params / surrogate RD) — **sans changer le format**.
> 🔁 **Monopackage** : `pc15` ré-exporte les fonctions clés (`encode_y`, `decode_y`, `CodecConfig`, `rans_*`, `read/write_bitstream`, etc.).

---

[**📚 Documentation en ligne**](https://agaloppe84.github.io/ProceduralCodec-v15/)

## Sommaire

- [Installation rapide (utilisateur)](#installation-rapide-utilisateur)
- [Installation (dev)](#installation-dev)
- [Philosophie & architecture](#philosophie--architecture)
- [Mini-guide d’utilisation](#mini-guide-dutilisation)
- [Stratégie IA (v15.1+)](#stratégie-ia-v151)
- [Workflows recommandés](#workflows-recommandés)
- [Releases & CI](#releases--ci)
- [Contribuer](#contribuer)
- [Licence](#licence)
- [Delta checklist (à garder aligné)](#delta-checklist-à-garder-aligné)

---

## Installation rapide (utilisateur)

> **Pré-requis** : Python ≥ 3.10.
> Sur Colab/CPU, pas besoin de CUDA pour décoder/tests (le rendu procédural GPU est recommandé mais optionnel pour la démo).

```bash
pip install "git+https://github.com/agaloppe84/ProceduralCodec-v15@v15.0.4"
````

Utilisation “one-namespace” :

```python
import pc15 as pc
from pc15 import CodecConfig  # exporté au top-level

cfg = CodecConfig(tile=256, overlap=24, lambda_rd=0.015, alpha_mix=0.7, seed=1234)
enc = pc.encode_y(img_y, cfg)             # -> {"bitstream": bytes, "bpp": float, "stats": dict, ...}
img_hat = pc.decode_y(enc["bitstream"])
```

> Astuce : `pc15.build_rans_tables`, `pc15.rans_encode`, `pc15.read_bitstream`, … sont aussi ré-exportés.

---

## Installation (dev)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r configs/requirements.dev.txt

# Dev monorepo (editable) si vous travaillez sur les sous-modules
pip install -e packages/pc15core -e packages/pc15proc -e packages/pc15codec \
            -e packages/pc15vq   -e packages/pc15metrics -e packages/pc15data \
            -e packages/pc15viz  -e packages/pc15wf

# Sans GPU local ?
export PC15_ALLOW_CPU_TESTS=1
pytest -q
```

**CUDA** : recommandé pour l’encodage ; PyTorch 2.x.

---

## Philosophie & architecture

### Encode (Y puis CbCr 4:2:0)

1. **Tiling** : découpe `(H,W)` en tuiles `T×T` avec overlap `O`.
2. **Candidats** : batch GPU de rendus procéduraux (générateur + params + seed).
3. **Score RD** : `D = α·(1-SSIM) + (1-α)·MSE_norm` + `λ·bits` (estimés → mesurés).
4. **Sélection / beam** : top-1 (ou beam K=2–4), écriture du **record tuile**.
5. **Entropie** : rANS sur symboles (IDs géné, indices QV, offsets, seed, flags).
6. *(Optionnel)* résidu parcimonieux si tuile “hors-classe”.
7. **Chroma 4:2:0** après la luminance.

### Decode

1. Lire **header** `PC15` + tables/codebooks.
2. Pour chaque tuile : décoder `(gen_id, qv_id|indices, seed, flags)` puis **re-rendre**.
3. **Blend** des tuiles (fenêtre Hann/cosine) sur l’overlap.

---

## Mini-guide d’utilisation

### 1) Procédural (`pc15proc`)

```python
import torch
from pc15proc import list_generators, render

print(list_generators()[:10])
img = render(
    name="STRIPES", H=256, W=256,
    params={"freq": 6.0, "angle_deg": 15.0, "phase": 0.25, "contrast": 0.9},
    batch=1, device=("cuda" if torch.cuda.is_available() else "cpu"), dtype=torch.float16,
)  # -> Tensor [1,1,256,256] in [-1,1]
```

### 2) Encode/Décode Y

```python
from pc15 import encode_y, decode_y, CodecConfig
cfg = CodecConfig(tile=256, overlap=24, lambda_rd=0.015, alpha_mix=0.7, seed=1234)
enc = encode_y(img_y, cfg)
bitstream, bpp = enc["bitstream"], enc["bpp"]
img_hat = decode_y(bitstream)
```

### 3) Métriques

```python
from pc15 import psnr, ssim
print(psnr(img_y, img_hat), ssim(img_y, img_hat))
```

### 4) Bitstream helpers

```python
from pc15 import read_bitstream, write_bitstream, rans_encode, rans_decode, build_rans_tables
```

---

## Stratégie IA (v15.1+)

> **Objectif** : accélérer/améliorer la **recherche** tout en **gardant le format identique**.
> L’IA ne touche **pas** au décodeur (bitstream inchangé). Elle agit **uniquement** côté encodeur.

### Slots IA optionnels (AiHooks)

1. **M1 — Tile Selector** : prédire top-K générateurs plausibles par tuile (softmax sur {génés}).
   **But** : ne rendre que K candidats pertinents → **×(2–10)** de gain selon *K* et la scène.

2. **M2 — Param Init** : prédire `(qv_id, offsets)` initiaux par géné.
   **But** : centrer la grille locale → **moins d’itérations** (coarse→fine plus court).

3. **M3 — RD Surrogate** : estimer `\^RD` (ou `\^D`/`\^bits`) pour ordonner/écrémer avant le score exact.
   **But** : calculer “cher” uniquement sur le **top-M** candidates.

**Pipeline** : image → tuiles → *features* → M1/M2/M3 → **search** (coarse→fine/beam) → rANS → bitstream.
**Fallback** : si `AiHooks=None`, l’encodeur fonctionne **comme aujourd’hui** (grilles par défaut).

### API (concept — v15.1)

```python
from pc15 import CodecConfig, encode_y
# from pc15core.ai import AiHooks, Selector, ParamPredictor, RdPredictor  # stubs v15.1 (à venir)

hooks = None  # si non fourni => chemin actuel
# hooks = AiHooks(
#     selector=Selector(weights="m1_selector_v0.pt"),
#     param_predictor=ParamPredictor(weights="m2_params_v0.pt"),
#     rd_predictor=RdPredictor(weights="m3_rd_v0.pt"),
# )

cfg = CodecConfig(tile=256, overlap=24, lambda_rd=0.015, alpha_mix=0.7, seed=1234, ai=hooks)
enc = encode_y(img_y, cfg)
```

### Rôle des modèles

**M1 — Selector (top-K génés)**

* *Entrée* : features de tuile (intensité normalisée, gradients, histos, pyramide multiscale).
* *Sortie* : `p(gen)` sur la banque de générateurs.
* *Perte* : CrossEntropy.
* *Usage* : on ne rend que les `K` meilleurs ➜ gros speed-up.

**M2 — ParamPredictor (init QV+offsets)**

* *Entrée* : patch + `gen_id`.
* *Sortie* : `(qv_id, offsets)` (offsets petits réels par dimension).
* *Pertes* : CE pour `qv_id`, L1/Huber pour offsets.
* *Usage* : centre la grille locale ➜ moins d’itérations.

**M3 — RdPredictor (surrogate RD)**

* *Entrée* : features + `(gen, params_init)`
* *Sortie* : `\^RD` (ou `\^D` et `\^bits`)
* *Perte* : L1/HMSE ; calibration isotone possible.
* *Usage* : tri/écrémage ➜ on calcule “exact” seulement sur le top-M.

### Données & préparation (hors-ligne)

1. **Extraction tuiles**

   ```bash
   python tools/pc15learn/make_tiles_jsonl.py \
     --images /path/to/photos_v13 \
     --out datasets/tiles.jsonl \
     --tile 256 --overlap 24
   ```
2. **Features** : (HOG/grad, histos, pyramide), normalisation **gelée** (stats du train).
3. **Labels** : oracle (recherche exhaustive sur sous-set) ou heuristique “teacher” v15.0.
4. **Split** : train/val/test **par image** (pas par tuile).
5. **Export** : `*.pt` + YAML de méta (versions, normalisation, seed).

### Déploiement

* Charger les poids en CPU/GPU.
* Renseigner `CodecConfig(ai=hooks)` ; écrire `header.meta.ai = {name, version, norm, sha256}`.
* Implémenter un **graceful fallback** (si poids manquants → ignorer le hook, warning).

### Bench & objectifs

* **Temps** : viser ×3 sur l’encodage Y avec M1 seul ; ×5–×8 avec M1+M2 ; M3 selon budget.
* **Qualité** : ΔPSNR / ΔSSIM ≤ 0.1–0.2 dB / 0.001 vs recherche brute (budget égal).
* **Bitrate** : Δbpp ≤ 1–2 % sur corpus v13.

### Roadmap IA

* **v15.1** : stubs `AiHooks` + interfaces, heuristiques par défaut, extraction de tuiles.
* **v15.2** : M1 entraîné (banque de génés v15), ablation sur *K*.
* **v15.3** : M2 offsets + M3 surrogate RD ; benchmarks temps×qualité ; publication poids baseline.

---

## Workflows recommandés

1. **bootstrap_online (CPU)** : venv + install, log versions → `reports/env_online_15.txt`
2. **proc_core_smoke (GPU)** : `pc15proc.render` 2–3 génés, previews `outputs/`
3. **encode_batch (GPU)** : `.pc15` idempotents + stats/CSV
4. **metrics_rd (CPU/GPU)** : PSNR/SSIM + RD plots
5. **publish_models (CPU)** : tables rANS, codebooks QV, configs figées dans *Models*

---

## Releases & CI

* **Release** :

  ```bash
  git tag v15.0.4
  git push origin v15.0.4
  ```

  ➜ déclenche le job **release** si le workflow cible `on.push.tags`.

* **Éviter les doubles builds** :
  Dans `ci.yml`, choisis **soit** `on.push.branches`, **soit** `on.push.tags` (pas les deux dans le même workflow).
  Alternative : séparer `ci.yml` (branches/PR) et `release.yml` (tags uniquement).

---

## Contribuer

* **Tests** : `pytest -q` (fallback CPU via `PC15_ALLOW_CPU_TESTS=1`).
* **Style** : black + ruff ; type hints.
* **PR** : inclure un test couvrant la surface API **et** le déterminisme (bitstream stable).

---

## Licence

À définir (`LICENSE`).

---

## Delta checklist (à garder aligné)

* **Packaging**

  * `pyproject.toml` : pin NumPy compatible envs (ex. Colab Py3.12 : `numpy>=1.26,<2.0` OK) — ou matrice conditionnelle.
  * `pc15/__init__.py` : ré-export **`CodecConfig`** + helpers (`encode_y`, `decode_y`, `rans_*`, `read/write_bitstream`, `score_rd_numpy`).
  * `__version__` = `15.0.4` cohérent avec le tag et les logs.

* **CI**

  * Un workflow pour branches/PR, un pour tags (optionnel) ; éviter déclenchements doublons.
  * Artefacts utiles : `pytest-report.xml`, audits génés, RD plots.

* **Docs**

  * Ce README couvre l’IA et l’API actuelle (single-package).
  * (Facultatif) Lien vers un notebook Colab démonstration.

* **Outils IA (à venir)**

  * `tools/pc15learn/make_tiles_jsonl.py` (extraction tuiles/labels).
  * `pc15core/ai.py` : classes `AiHooks`, `Selector`, `ParamPredictor`, `RdPredictor` (stubs no-op si poids absents).


Parfait — voilà **les 2 blocs prêts à coller** dans ton README.

---

# Bloc README — Colab Quickstart (API + Drive + Hooks)

```python
#@title PC15 — Colab Quickstart (API + Drive + Hooks)
REPO = "agaloppe84/ProceduralCodec-v15"  # branche/tag via TAG
TAG  = "main"                            # ex: "v15.0.4" pour une release

import sys, subprocess, os, pathlib, time, json, math
PY = sys.executable
def sh(*args): print("➜", " ".join(map(str,args))); subprocess.run(list(args), check=True)

# 1) Dépendances minimales (+ numpy pin Colab)
sh(PY, "-m", "pip", "install", "-q",
   "numpy>=1.26,<2.0", "Pillow>=10.0", "tqdm>=4.66", "matplotlib>=3.7")

# 2) Installer PC15 (monopackage avec sous-modules intégrés)
sh(PY, "-m", "pip", "install", "-q", f"git+https://github.com/{REPO}@{TAG}")

# 3) Monter Google Drive (on n’utilise QUE Drive ici)
from google.colab import drive
drive.mount("/content/drive", force_remount=True)

# 4) Config des chemins (tout passe par PathsConfig)
import pc15 as pc
print("pc15 version:", getattr(pc, "__version__", "unknown"))

DATASET_IMAGES = "/content/drive/MyDrive/procedural_datasets/pc_synth_v1_plus/images"  # <-- ton dataset (2000 photos)
BASE_DIR       = "/content/drive/MyDrive/pc15"                                         # répertoire racine PC15 (artifacts, cache…)

paths = pc.PathsConfig(
    base=BASE_DIR,
    dataset_images=DATASET_IMAGES,     # images d'entraînement/étiquetage
    artifacts_subdir="artifacts",      # ex: /content/drive/MyDrive/pc15/artifacts
    cache_subdir=".cache",             # ex: /content/drive/MyDrive/pc15/.cache
    labels_dir=None,                   # (optionnel) ex: "/content/drive/MyDrive/pc15/labels"
)
print("ENV:", pc.env_summary())

# 5) Smoke encode/decode (Y) + comparaison Hooks heuristiques vs baseline
import numpy as np
def _to_tensor(a):
    import torch
    return torch.as_tensor(a, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

H=W=256
grad = np.linspace(0,1,W, dtype=np.float32)[None,:].repeat(H,axis=0)*0.8+0.1
yy   = _to_tensor(grad)

cfg = pc.CodecConfig(tile=256, overlap=24, lambda_rd=0.015, alpha_mix=0.7, seed=1234)

t0=time.perf_counter(); enc0 = pc.encode_y(yy, cfg); t_enc0=(time.perf_counter()-t0)*1000
t1=time.perf_counter(); recon= pc.decode_y(enc0["bitstream"], device="cpu"); t_dec=(time.perf_counter()-t1)*1000

# Hooks non-réseau de neurones (ex: top-K heuristique)
hooks = pc.HeuristicHooks(topk=8)
t2=time.perf_counter(); enc1 = pc.encode_y_with_hooks(yy, cfg, hooks=hooks); t_enc1=(time.perf_counter()-t2)*1000

# PSNR/SSIM (fallback NumPy si pc15.metrics indispo)
def _as_np(x):
    import torch
    return x.detach().cpu().float().squeeze().numpy() if isinstance(x, torch.Tensor) else np.asarray(x, np.float32)

try:
    PSNR = float(pc.psnr(yy, recon).item()); SSIM=float(pc.ssim(yy, recon).item())
except Exception:
    a=_as_np(yy); b=_as_np(recon)
    mse=float(np.mean((a-b)**2)); PSNR=20*math.log10(1.0)-10*math.log10(max(mse,1e-8))
    SSIM=max(0.0, 1.0-mse/(float(np.mean(a**2))+1e-8))

print(f"Baseline encode: {t_enc0:.1f} ms  | bytes={len(enc0['bitstream'])}")
print(f"Hooks encode   : {t_enc1:.1f} ms  | bytes={len(enc1['bitstream'])}")
print(f"Decode         : {t_dec:.1f} ms   | PSNR={PSNR:.2f} dB  SSIM={SSIM:.4f}")

# 6) Écriture d’artefacts sur Drive
LOG_DIR = pathlib.Path(paths.base)/"artifacts"; LOG_DIR.mkdir(parents=True, exist_ok=True)
HTML = LOG_DIR/"pc15_report.html"
HTML.write_text(
    f"<html><body><h1>PC15 Colab</h1>"
    f"<p>pc15={getattr(pc,'__version__','?')}</p>"
    f"<p>Baseline bytes={len(enc0['bitstream'])} | Hooks bytes={len(enc1['bitstream'])}</p>"
    f"<p>Encode0={t_enc0:.1f} ms | Encode1={t_enc1:.1f} ms | Decode={t_dec:.1f} ms</p>"
    f"<p>PSNR={PSNR:.2f} dB | SSIM={SSIM:.4f}</p></body></html>", encoding="utf-8"
)
print("Saved:", HTML)
```

---

# Bloc README — Arborescence standard pour tout **nouvel outil** intégré

> Règle d’or : **jamais de code d’outil à la racine**.
> Chaque outil vit dans **`packages/pc15<outil>/src/pc15<outil>/...`** et le *namespace agrégateur* `pc15` ré-exporte ce qu’il faut.

```
# Racine du repo
.
├─ pyproject.toml                 # unique (build setuptools) — liste les src des sous-packages
├─ src/
│  └─ pc15/
│     ├─ __init__.py              # agrégateur : importe/alias les sous-packages et ré-exporte l'API
│     └─ ...                      # (facultatif) petites façades communes
├─ packages/
│  ├─ pc15codec/                  # existant
│  │  └─ src/pc15codec/...
│  ├─ pc15proc/                   # existant
│  │  └─ src/pc15proc/...
│  ├─ pc15metrics/                # existant
│  │  └─ src/pc15metrics/...
│  ├─ pc15data/                   # existant
│  │  └─ src/pc15data/...
│  ├─ pc15viz/                    # existant
│  │  └─ src/pc15viz/...
│  ├─ pc15wf/                     # existant
│  │  └─ src/pc15wf/...
│  ├─ pc15vq/                     # existant
│  │  └─ src/pc15vq/...
│  └─ pc15learn/                  # **exemple d’outil intégré** (ex-tools)
│     └─ src/
│        └─ pc15learn/
│           ├─ __init__.py        # expose l’API de l’outil (ex: HeuristicHooks, PathsConfig, make_labels, …)
│           ├─ paths.py           # gestion des chemins paramétriques (Colab/Drive/locaux)
│           ├─ ai_hooks.py        # hooks heuristiques + encode_y_with_hooks(...)
│           └─ make_labels.py     # CLI/func d’extraction de tuiles/labels (v15.1+)
├─ tests/
│  ├─ test_api_contracts.py
│  ├─ test_pc15learn_hooks.py     # tests du nouvel outil (import via `from pc15.learn import ...`)
│  └─ test_pc15learn_labels.py
└─ ...
```

**Checklist d’intégration d’un nouvel outil `pc15<outil>` :**

1. **Dossier** : `packages/pc15<outil>/src/pc15<outil>/...` (pas de code outil ailleurs).

2. **`pyproject.toml`** (racine) → ajouter le chemin dans :

   ```toml
   [tool.setuptools.packages.find]
   where = [
     "src",
     "packages/pc15core/src", "packages/pc15proc/src", "packages/pc15codec/src",
     "packages/pc15vq/src", "packages/pc15metrics/src", "packages/pc15data/src",
     "packages/pc15viz/src", "packages/pc15wf/src",
     "packages/pc15learn/src"        # <-- ajouter votre nouvel outil ici
   ]
   ```

   *(Remplace `pc15learn` par `pc15<outil>` pour le prochain outil.)*

3. **Agrégateur `src/pc15/__init__.py`** → alias propre :

   ```python
   # Exemple pour un outil 'pc15learn'
   try:
       import pc15learn as learn
       from pc15learn import HeuristicHooks, PathsConfig, encode_y_with_hooks, env_summary, make_labels
   except Exception:
       learn = None
       HeuristicHooks = PathsConfig = encode_y_with_hooks = env_summary = make_labels = None  # type: ignore

   __all__ += [
       "learn", "HeuristicHooks", "PathsConfig", "encode_y_with_hooks", "env_summary", "make_labels"
   ]
   ```

   *Même pattern pour tout `pc15<outil>` futur (ex: `pc15train`, `pc15serve`, …).*

4. **Tests** : placer les tests à la racine `tests/` et importer via **`from pc15.<alias> import ...`**
   (on ne référence jamais `packages/...` directement dans les tests).

5. **Docs** : documenter l’API dans le README principal (sections *Quickstart/Colab* et *Arborescence*).

C’est tout — avec ça, **un seul `pip install pc15`** expose l’outil **et** l’agrégateur `pc15` ré-exporte l’API utile.


## Step 0 — Pré-flight & gels (v15)

- **CodecConfig** unique (tile/overlap/λ/α/rans_id) — `from pc15codec.config import CodecConfig`.
- **Tables rANS gelées** packagées sous `pc15codec.data.rans` + override via `PC15_MODELS_DIR/rans/<id>.json`. <!-- [ML] -->
- **Header v15 + CRC32** (`pack_v15`/`unpack_v15`) — framing stable.
- **Seed policy** déterministe (`pc15codec.seed.tile_seed`) — idempotence stricte.
- **Bitstreams I/O** atomique (`read_bitstream` / `write_bitstream`). <!-- [STORE:OVERWRITE] -->
- **Chemins paramétriques** (`PathsConfig`) + `outputs_path`/`artifacts_path`. <!-- [STORE:OVERWRITE]/[STORE:CUMULATIVE] -->
- **Tests** OK : public_api, header_crc, seed_policy, rans_tables (+ *futureproof* skips).

Parfait — voilà le **bloc README Step 1** mis à jour, incluant les derniers correctifs (indices sans overshoot + crop au bord), les fenêtres **Hann/Tukey/Kaiser**, et le **seam penalty**. Tu peux **copier/coller** tel quel dans ton `README.md`.

---

## Step 1 — Tiling & Blend + Tile Records (v15)

### TL;DR

* **Tiling** paramétrique (grille robuste, **sans overshoot**).
* **Blend** par *partition of unity* (normalisation par somme des poids) avec fenêtres **Hann** *(défaut)*, **Tukey** *(recommandée)*, **Kaiser**.
* **Crop automatique** au bord pour les tuiles partielles (pas d’erreur de taille).
* **Seam penalty mask** pour pénaliser légèrement les coutures dans le score RD.
* **TileRec** (format d’enregistrement de tuile, sans rANS pour l’instant).

---

### API (import)

```python
from pc15codec.tiling import TileGridCfg, tile_image, blend, seam_penalty_mask
from pc15codec.bitstream import TileRec
```

---

### Tiling (grille)

```python
grid = TileGridCfg(size=256, overlap=24)    # vérifs intégrées : size>0 et 0<=overlap<size
spec = tile_image(y, grid)                  # y: Tensor[1,1,H,W]
# spec: TileBatchSpec(H, W, size, overlap, ny, nx, starts) + .count
```

**Invariants & garanties**

* Les positions de départ sont **monotones** et **ne dépassent jamais** `L - size`.
* La dernière tuile est **forcée** à `L - size` pour couvrir le bord.
* `spec.starts` est de longueur `spec.count = spec.ny * spec.nx`.

---

### Blend (partition of unity)

```python
y_hat = blend(
    tiles, spec, H, W,                 # tiles: Tensor[N,1,size,size], N == spec.count
    window="hann",                     # "hann" (défaut) | "tukey" | "kaiser"
    window_params=None                 # {'alpha':...} pour tukey, {'beta':...} pour kaiser
)
```

**Comportement**

* Accumulation pondérée `sum(tiles * w)` puis **normalisation** par `sum(w)` pixel-par-pixel.
* **Crop automatique** de la fenêtre et de la tuile quand une tuile touche le bord (aucune exception sur les tailles).
* Fenêtres 2D **séparables** (wy⊗wx) pour robustesse et perf.

**Fenêtres**

* **Hann** *(défaut)* : simple, fiable.
* **Tukey** *(recommandée prod)* : plateau central + bords lissés.

  * `alpha` **auto** ≈ `2 * overlap / size` (borné à [0,1]).
  * Exemple : `blend(..., window="tukey")` ou `blend(..., window="tukey", window_params={"alpha":0.5})`.
* **Kaiser** : bords plus “raides” (textures très structurées).

  * `beta` typique `6.5–8.0`.
  * Exemple : `blend(..., window="kaiser", window_params={"beta":6.5})`.

---

### Seam penalty (RD)

```python
m = seam_penalty_mask(size=grid.size, overlap=grid.overlap, power=1.5)  # Tensor[size,size] ∈ [0,1]
# Exemple d’intégration dans le score RD :
# D_base = α·(1-SSIM) + (1-α)·MSE_norm
# D = (1 + γ·m) * D_base    # γ ≈ 0.25…0.5 selon la sensibilité aux coutures
```

* `m≈0` au centre, `→1` vers les bords ; `power>1` accentue la pénalité près des edges.
* **But** : éviter que la recherche sélectionne des candidats “bons au centre mais mauvais sur les coutures”.

> 💬 **Commentaire stockage/IA** : ce masque est **calculé à la volée** (pas de stockage). La pondération RD ne stocke rien non plus. *(Aucun artefact IA/entrainement ici.)*

---

### Tile records (bitstream)

```python
rec = TileRec(tile_id=0, gen_id=7, qv_id=3, seed=42,
              rec_flags=0, payload_fmt=0, payload=b"")   # pas de rANS encore
d = rec.to_dict(); rec2 = TileRec.from_dict(d)           # round-trip dict
```

* **Objectif Step 1** : typer la structure de record par tuile.
* **Pas de sérialisation rANS** encore (payload permissif, éventuellement vide).
* **Stockage** : quand on activera rANS, ce sera un bloc **[STORE:OVERWRITE]** packé dans le bitstream. Ici, rien n’est écrit.

---

### Exemples

**Hann (défaut)**

```python
H, W = 512, 512
y     = torch.zeros((1,1,H,W), device=dev, dtype=dtype)
grid  = TileGridCfg(size=128, overlap=16)
spec  = tile_image(y, grid)
tiles = torch.rand((spec.count, 1, grid.size, grid.size), device=dev, dtype=dtype)

y_hat = blend(tiles, spec, H, W)   # partition of unity, crop auto au bord
```

**Tukey (prod)**

```python
y_hat = blend(tiles, spec, H, W, window="tukey")  # alpha auto en fonction overlap/size
```

**Kaiser**

```python
y_hat = blend(tiles, spec, H, W, window="kaiser", window_params={"beta": 6.5})
```

---

### Edge cases & perfs

* **Bords** : pas de mismatch — crop appliqué sur (tuile, fenêtre) au besoin.
* **CPU/GPU** : supportés ; `float16` en CUDA, `float32` en CPU.
* **Stride** = `size - overlap` (implémenté via indices monotones).
* **Aucun I/O** ici (pas de disque), tout est in-mem.

---

### Tests (CI)

* ✅ `test_step1_tiling_blend_shapes.py` — formes, finitude, dynamique [0..1].
* ✅ `test_step1_tilerec_roundtrip.py` — round-trip dict `TileRec`.
* ✅ `test_api_contracts.py` — micro-pipeline STRIPES + tiling/blend.
* 🔜 Tests rANS/payload restent **skipped** jusqu’au Step 2.

> Pour forcer les tests CPU : `PC15_ALLOW_CPU_TESTS=1` (déjà dans le workflow).
> **PyTorch CPU 2.3.1** OK ; `kaiser_window` dispose d’un fallback interne si absent.

---

### Roadmap (prochain bloc)

* **Step 2 — Bitstream records** : sérialisation multi-tuiles, payload rANS (tables gelées v15 + override), read/write des records avec CRC, invariants + tests round-trip.
* **Step 3 — Recherche RD** : boucle coarse→fine + seam penalty intégrée, beam K (optionnel), early-exit, métriques PSNR/SSIM mixées.

---
