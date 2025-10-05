[![CI](https://github.com/agaloppe84/ProceduralCodec-v15/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/agaloppe84/ProceduralCodec-v15/actions/workflows/ci.yml)

# ProceduralCodec v15 — Codec procédural par tuiles (PC15)

**PC15** est un codec d’image **entièrement procédural** : l’image est un **programme** (suite de générateurs + paramètres quantifiés + seeds). L’encodage sélectionne, tuile par tuile (GPU), le meilleur couple *(générateur, paramètres, seed)* via un score **RD** (distorsion + coût en bits) et produit un bitstream **déterministe et idempotent**.

---

## TL;DR

* **Tuiles GPU** (ex. 256×256, overlap 16–32), blending Hann.
* **Recherche** coarse→fine + early-exit, seeds int64, **QV/VQ** pour params corrélés.
* **Bitstream v15** : header `PC15`, tables rANS, codebooks QV, records par tuile, checksum.
* **APIs** (8 paquets) : `pc15core`, `pc15proc`, `pc15codec`, `pc15vq`, `pc15metrics`, `pc15data`, `pc15viz`, `pc15wf`.
* **Perf** : PyTorch 2.x, AMP fp16 si possible, `channels_last`, cuDNN autotune.

---

## Installation (dev)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r configs/requirements.dev.txt
pip install -e packages/pc15core -e packages/pc15proc -e packages/pc15codec \
            -e packages/pc15vq   -e packages/pc15metrics -e packages/pc15data \
            -e packages/pc15viz  -e packages/pc15wf

# Si pas de GPU local
export PC15_ALLOW_CPU_TESTS=1
pytest -q
```

**CUDA** : recommandé pour l’encodage ; PyTorch 2.3+.

---

## Philosophie & architecture

### Encode (Y puis CbCr 4:2:0)

1. **Tiling**: découpe `(H,W)` en tuiles `T×T` avec overlap `O`.
2. **Candidats**: batch GPU de rendus procéduraux (générateur + params + seed).
3. **Score RD**: `D = α·(1-SSIM) + (1-α)·MSE_norm` + `λ·bits_estimés`.
4. **Sélection / beam**: top‑1 (ou beam K=2–4), écriture du **record tuile**.
5. **Entropie**: rANS sur symboles (IDs géné, indices QV, offsets, seed, flags).
6. *(Optionnel)* résidu parcimonieux si tuile hors‑classe.
7. **Chroma 4:2:0** après la luminance.

### Decode

1. Lire **header** `PC15` + tables/codebooks.
2. Pour chaque tuile : décoder `(gen_id, qv_id|indices, seed, flags)` puis **re‑rendre**.
3. **Blend** les tuiles (fenêtre Hann/cosine) sur l’overlap.

---

## Mini‑guide d’utilisation

### 1) Procédural (`pc15proc`)

Lister et rendre un motif procédural :

```python
import torch
from pc15proc import list_generators, render

print(list_generators()[:10])
img = render(
    name="STRIPES",
    H=256, W=256,
    params={"freq": 6.0, "angle_deg": 15.0, "phase": 0.25, "contrast": 0.9},
    batch=1, device="cuda", dtype=torch.float16,
)  # Tensor [1,1,256,256] dans [-1,1]
```

Paramètres typés via `ParamCodec` (int/float/enum/bool), avec quantification en QV (voir `pc15vq`).

### 2) Tiling & blend (`pc15codec`)

```python
from pc15codec.tiling import TileGridCfg, tile_image, blend_tiles

cfg = TileGridCfg(tile=256, overlap=24)
patches, grid = tile_image(img_y, cfg)      # img_y: Tensor [1,1,H,W]
# ... faire tourner la recherche et choisir un synthé par tuile ...
recon = blend_tiles(patches_hat, grid, cfg) # Tensor [1,1,H,W]
```

### 3) Encode/Décode Y (bitstream v15)

```python
from pc15codec import encode_y, decode_y

out = encode_y(img_y, cfg={"tile":256, "overlap":24, "lambda":0.015, "alpha":0.7})
bitstream, bpp, stats = out["bitstream"], out["bpp"], out["stats"]
img_hat = decode_y(bitstream)
```

`stats` agrège par générateur/tuile : distorsion, bits estimés, early‑exits, etc.

### 4) Métriques (`pc15metrics`)

```python
from pc15metrics import psnr, ssim
print(psnr(img_y, img_hat), ssim(img_y, img_hat))
```

### 5) Données & manifestes (`pc15data`)

```python
from pc15data import scan_images, to_luma_tensor, build_manifest
imgs = scan_images("/path/to/images")
img_y = to_luma_tensor(imgs[0])        # [1,1,H,W] float in [-1,1]
build_manifest(imgs, "manifests/run_001.json")
```

### 6) VQ / Quantification vectorielle (`pc15vq`)

```python
from pc15vq import train_qv, quantize_params
codebook = train_qv(samples, dims=3, k=256)
indices, offsets = quantize_params(params_batch, codebook)
```

### 7) Viz (`pc15viz`)

```python
from pc15viz import montage, plot_rd
montage_img = montage(u8_batch, cols=8)       # PIL.Image
plot_rd("artifacts/rd.csv", "artifacts/rd.png")
```

### 8) Orchestration (`pc15wf`)

```python
from pc15wf import pc15_name, load_manifest, log_append
name = pc15_name("photos_v1", "000123", "codec_v15", q=75, bpp=0.45, seed=1234)
log_append("reports/run_2025-10-05.txt", {"event": "encode_done", "name": name})
```

---

## Workflows recommandés

1. **bootstrap_online (CPU)**

   * Créer venv, installer `requirements.dev.txt` (sans torch GPU), loguer versions → `reports/env_online_15.txt`.
2. **proc_core_smoke (GPU)**

   * `pc15proc.render` sur 2–3 génés, save previews `outputs/` (sanity check GPU, seeds déterministes).
3. **encode_batch (GPU)**

   * Encoder un manifeste d’images (Y), produire `.pc15` (idempotent, noms invariants), écrire stats/CSV.
4. **metrics_rd (CPU/GPU)**

   * Décoder, calculer PSNR/SSIM, agrégats CSV/JSON, tracer RD.
5. **publish_models (CPU)**

   * Publier tables **rANS**, **codebooks QV** et **configs** figées dans *Models*.

---

## CLI & Examples

> **Pré-requis** : environnement activé, paquets installés en editable (voir section *Installation*), et un GPU CUDA pour l’encodage.

### Exemples rapides (one‑liners)

**1) Rendu procédural → PNG (sanity check GPU)**

```bash
python - <<'PY'
import torch
from pc15proc import render
from PIL import Image

y = render("STRIPES", 256, 256,
           {"freq": 6.0, "angle_deg": 15.0, "phase": 0.25, "contrast": 0.9},
           batch=1, device=("cuda" if torch.cuda.is_available() else "cpu"), dtype=torch.float16)[0,0]
# Tensor [-1,1] -> u8 [0,255]
u8 = ((y.clamp(-1,1)+1)*127.5).byte().cpu().numpy()
Image.fromarray(u8, mode="L").save("outputs/stripes.png")
print("✓ wrote outputs/stripes.png")
PY
```

**2) Encode→Decode d’une image Y (API directe)**

```bash
python - <<'PY'
import torch, sys
from pathlib import Path
from PIL import Image
from pc15data import to_luma_tensor
from pc15codec.tiling import TileGridCfg
from pc15codec import encode_y, decode_y

inp = sys.argv[1] if len(sys.argv)>1 else "tests/assets/lena.png"
img_y = to_luma_tensor(inp)                               # [1,1,H,W] in [-1,1]
cfg = {"tile":256, "overlap":24, "lambda":0.015, "alpha":0.7}
enc = encode_y(img_y, cfg)
bitstream, bpp = enc["bitstream"], enc["bpp"]
print(f"bpp={bpp:.3f}")
img_hat = decode_y(bitstream)
# Sauvegarde PNG
u8 = ((img_hat[0,0].clamp(-1,1)+1)*127.5).byte().cpu().numpy()
Path("outputs").mkdir(exist_ok=True, parents=True)
Image.fromarray(u8, mode="L").save("outputs/recon.png")
print("✓ wrote outputs/recon.png")
PY
```

> Astuce : `CUDA_VISIBLE_DEVICES=0` pour choisir le GPU ; `PIP_CACHE_DIR=/storage/.cache/pip` pour accélérer les installs en environnement distant.

### CLIs (référence)

Les commandes ci‑dessous correspondent aux scripts d’**orchestration** attendus dans `app/`. Elles servent d’exemples reproductibles ; adapte les chemins/flags à ton infra.

**1) Audit des générateurs**

```bash
python app/audit_generators.py \
  --out artifacts
# Produit artifacts/audit_generators.(csv|json)
```

**2) Encodage batch à partir d’un dossier d’images**

```bash
python app/encode_batch.py \
  --images /path/to/images \
  --tile 256 --overlap 24 \
  --lambda 0.015 --alpha 0.7 \
  --out outputs --manifest manifests/run_001.json \
  --seed 1234
```

**3) Décodage d’un bitstream .pc15**

```bash
python app/decode.py \
  --bitstream outputs/example.pc15 \
  --out outputs/example_recon.png
```

**4) Metrics + courbe RD**

```bash
python app/metrics.py \
  --manifest manifests/run_001.json \
  --out artifacts --rd-png artifacts/rd.png
```

**5) Vérification de déterminisme / idempotence**

```bash
python app/bitstream_diff.py A.pc15 B.pc15
# Exit code 0 si byte-à-byte identiques, ≠0 sinon (affiche 1er offset qui diffère)
```

> Si tu ne souhaites pas maintenir des scripts dans `app/`, expose des **entry points** pip (`console_scripts`) :
> `pc15-audit`, `pc15-encode`, `pc15-decode`, `pc15-metrics`, `pc15-bsdiff` pointant vers les mêmes fonctions.

---

## Règles de déterminisme & idempotence

* **Seeds** : int64 signés, dérivations par *slot* stables.
* **Noms** : incluent dataset, image, codec, `v15`, Q/λ, **bpp**, seed.
* **Écriture atomique** : `.tmp` → `os.replace` ; skip si fichier valide existe.
* **Logs** : append‑only sous `reports/` ; manifeste d’état `todo/done/error`.
* **Garde‑fous** : sans CUDA ⇒ `RuntimeError("Manque: GPU CUDA")`; symbole manquant ⇒ `RuntimeError("Manque: <symbole>")`.

---

## Arborescence projet (conseillée)

```
/storage/pc15/v15/
├─ configs/       # requirements.v15.txt, codec.json, search.json
├─ artifacts/     # CSV/JSON agrégés, figures RD, codebooks QV
├─ outputs/       # previews, reconstructions
├─ reports/       # logs append-only (ex: reports/run_YYYY-MM-DD.txt)
├─ manifests/     # runs, index tuiles, erreurs
├─ datasets/ -> /storage/datasets/pc15/photos_v1
└─ app/           # scripts d’orchestration (encode_batch.py, metrics.py, ...)
```

---

## Contribuer

* **Tests** : `pytest -q` (fallback CPU via `PC15_ALLOW_CPU_TESTS=1`).
* **Style** : black + ruff ; type hints.
* **PR** : inclure un test qui couvre la nouvelle surface API et maintient le déterminisme.

## Licence

À définir (`LICENSE`).

---

# Checklists par package

> **Principe** : pour chaque sous‑package, voici la surface API cible, une short‑list de tests à fournir, et des TODOs priorisés. Adapte selon l’état réel du code.

## `pc15proc` — Générateurs procéduraux GPU

**Surface API**

* `list_generators() -> list[str]`
* `render(name, H, W, params, batch=16, device="cuda", dtype=torch.float16) -> Tensor[B,1,H,W]`
* `ParamCodec` pour sérialiser/quantifier (int/float/enum/bool)

**Tests à écrire**

* Découverte/registry : unicité des noms, ≥ N génés.
* Seeds déterministes : même cfg ⇒ mêmes pixels.
* Dtypes : fp16/fp32 équivalents (tolérance ε).
* Perf smoke (GPU dispo) : batch × VRAM.

**TODO**

* [HAUT] Harmoniser les espaces de paramètres + ranges par géné.
* [HAUT] Ajouter pénalité *seam* (bord) dans le score local (si calculé ici).
* [MOYEN] Audit auto : CSV/JSON des génés (nom, params, ranges, has_noise/warp...).
* [BAS] Docstring par géné + preview dans README (montage).

---

## `pc15codec` — Tiling, RD search, bitstream

**Surface API**

* `tiling.TileGridCfg(tile:int, overlap:int)`
* `tile_image(img_y, cfg) -> (patches, grid)` ; `blend_tiles(patches_hat, grid, cfg) -> img_y_hat`
* `encode_y(img_y, cfg) -> {bitstream: bytes, bpp: float, stats: dict, tile_map: list}`
* `decode_y(bitstream, device="cuda") -> img_y_hat`

**Tests à écrire**

* Tiling/blend exact (identité, SSIM≈1 sur no‑op).
* Encode→decode déterministe (byte‑à‑byte bitstream stable).
* Modèle de bits : bpp cohérent (bornes raisonnables) sur patchs synthétiques.

**TODO**

* [HAUT] Implémenter rANS + tables stables (header/records/footer + CRC).
* [HAUT] Early‑exit + beam K (configurable) ; logs stats détaillées.
* [MOYEN] Résidu parcimonieux (optionnel, plafond de taille).
* [BAS] Support CbCr 4:2:0 complet + encode/décode chroma.

---

## `pc15vq` — Quantification vectorielle

**Surface API**

* `train_qv(samples, dims, k) -> codebook`
* `quantize_params(params_batch, codebook) -> (indices, offsets)`

**Tests à écrire**

* Reproductibilité (seed fixée) ; qualité de reconstruction (MSE/SSIM sur params synthétiques).
* Sérialisation/descripteur de codebook (pour header).

**TODO**

* [HAUT] Paquets sémantiques (ex. `[freq, angle, phase]`).
* [MOYEN] Stratégie multi‑résolution des params (coarse→fine).

---

## `pc15metrics` — Métriques & agrégats

**Surface API**

* `psnr(y, yhat)`, `ssim(y, yhat)`
* `summarize_batch(manifest_json, out_csv, out_json)`

**Tests à écrire**

* Identité : SSIM≈1/PSNR=∞ (borne) ; bruit gaussien : valeurs plausibles.

**TODO**

* [MOYEN] Mix `D = α·(1-SSIM) + (1-α)·MSE_norm` packagé.

---

## `pc15data` — Datasets & manifestes

**Surface API**

* `scan_images(root) -> list[Path]`
* `to_luma_tensor(path) -> Tensor[1,1,H,W]`
* `build_manifest(images, out_json)`
* `ensure_symlink(src, dst)`

**Tests à écrire**

* Lecture robustes de formats (PNG/JPG), normalisation [-1,1], tailles variées.

**TODO**

* [MOYEN] Cache mmap/LMDB (optionnel) pour IO massives.

---

## `pc15viz` — Visualisation

**Surface API**

* `montage(u8_batch, cols) -> PIL.Image`
* `plot_rd(csv_path, out_png)`
* `preview_tile_vs_synth(...)`

**Tests à écrire**

* Génération d’images (valide/non‑vide), tailles correctes.

**TODO**

* [BAS] Galerie README : spritesheet des generators.

---

## `pc15wf` — Orchestration & idempotence

**Surface API**

* `atomic_write(path, data)`
* `pc15_name(dataset, img, codec, q, bpp, seed) -> str`
* `load_manifest/save_manifest`
* `log_append`

**Tests à écrire**

* Écriture atomique (crash‑safety), re‑run idempotent (skip si OK), format nommage stable.

**TODO**

* [MOYEN] CLI `app/encode_batch.py`, `app/metrics.py` avec reprise `todo/done/error`.

---

## CI & Qualité

* **Matrix** CPU/GPU (si runner GPU possible) ; fallback CPU via `PC15_ALLOW_CPU_TESTS=1`.
* Artefacts : `artifacts/audit_generators.(csv|json)`, `artifacts/rd.png`, `pytest-report.xml`.
* Linting : `ruff`, `black`, `mypy` (optionnel).

---

## FAQ

* **Pourquoi des tuiles ?** Mémoire bornée + parallélisme + localité.
* **Pourquoi procédural ?** Décodeur sans apprentissage, déterminisme, bitstream compact.
* **Pourquoi rANS ?** Entropie efficace et tables partageables.
