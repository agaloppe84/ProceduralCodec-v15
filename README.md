````markdown
[![CI](https://github.com/agaloppe84/ProceduralCodec-v15/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/agaloppe84/ProceduralCodec-v15/actions/workflows/ci.yml)

# ProceduralCodec v15 — Codec procédural par tuiles (PC15)

**PC15** est un codec d’image **entièrement procédural** : l’image est un **programme** (suite de générateurs + paramètres quantifiés + seeds).
L’encodage sélectionne, **tuile par tuile** (GPU), le meilleur couple *(générateur, paramètres, seed)* via un score **RD** (distorsion + coût en bits) et produit un bitstream **déterministe et idempotent**.

> ✅ **État v15.0.4** : encode/décode **Y**, bitstream **v15** (rANS auto-portant), tiling/blend, métriques, orchestration.
> 🧠 **Pré-intégration IA (v15.1+)** : hooks *optionnels* côté encodeur (sélecteur de générateurs / init params / surrogate RD) — **sans changer le format**.
> 🔁 **Monopackage** : `pc15` ré-exporte les fonctions clés (`encode_y`, `decode_y`, `CodecConfig`, `rans_*`, `read/write_bitstream`, etc.).

---

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

```
::contentReference[oaicite:0]{index=0}
```
