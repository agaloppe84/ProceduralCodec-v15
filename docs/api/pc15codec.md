# pc15codec

## Step 3 - Payload ANS0 (rANS)

Format: `payload = b"ANS0" + u8 L + table_id[:L] + rANS_stream_bytes`
Tables gelées: `pc15codec.rans.DEFAULT_TABLE_ID = "v15_default"`
Override: `PC15_RANS_TABLES=/abs/path/with/json`

API:
```python
from pc15codec.payload import ANS0_FMT, encode_tile_payload, decode_tile_payload
fmt, blob = encode_tile_payload(gen_id, qv_id, seed, flags, offsets, table_id="v15_default")
g,qv,seed,flags,offs = decode_tile_payload(blob)
```

---

## Critères d’acceptation (Step 3)
- `write_stream_v15/read_stream_v15` **inchangés** (déjà OK). :contentReference[oaicite:10]{index=10}
- Les `.pc15` produits contiennent des `TileRec.payload_fmt == 0 (ANS0)` et se **décodent sans erreur** (round-trip symbolique).
- `encode_tile_payload` / `decode_tile_payload` **CPU-only** OK.
- Tables gelées chargées via `load_table_by_id("v15_default")`. :contentReference[oaicite:11]{index=11}
- Back-compat : `PC15_PAYLOAD_FMT=RAW` → chemin actuel **RAW** inchangé. :contentReference[oaicite:12]{index=12}

---

## Step 4 - Reconstruction & RD minimal

**But** : décoder les symboles ANS0 → (gen_id, qv_id, seed, flags, offsets) → **déquantifier** les paramètres → **rendre** la tuile (proc) → **blender** dans le canvas Y.
Côté encodage, une **recherche coarse** sélectionne par tuile le meilleur candidat au **score RD**.

### Pipeline

**Encode (par tuile)**
1. Génère un petit set de candidats (générateur + params “coarse”).
2. `dist = α·(1-SSIM) + (1-α)·MSE_norm` sur la tuile de référence.
3. `bits` estimés via tables rANS gelées (ou “exact” par encodage réel).
4. **Score** `RD = dist + λ·bits`; garde le meilleur.
5. `quantize_params` → `(qv_id, offsets)` ; `pack_symbols` ; `encode_tile_payload` (ANS0).

**Decode (par tuile)**
1. `decode_tile_payload` (lit `b"ANS0"|L|table_id|stream`).
2. `dequantize_params(qv_id, offsets)` → `params`.
3. `pc15proc.render(gen_name, h, w, params)` → tuile synthèse.
4. **Blend Hann** sur l’overlap → canvas final Y `[1,1,H,W]`.

---

### Critères d’acceptation - Step 4

- `decode_y` **reconstruit** une image **non nulle** (plus de noir).
- `encode_y` effectue une **recherche** minimale et choisit le meilleur **score RD** (coarse → pas encore de refine/beam).
- **Déterminisme** : deux encodes strictement identiques (mêmes cfg/seed/image) produisent **exactement les mêmes bytes**.
- `estimate_bits_from_table` (mode *table*) est **proche** des bits *exact* (mode *exact*) : tolérance ±8 bits ou ±5 %.
- QV : `quantize_params` / `dequantize_params` **opèrent** et conservent l’échelle globale des paramètres.
- **Compat v15** : framing inchangé (header/records/stream). Seuls les payloads passent en ANS0.

---

### Variables d’environnement utiles

| Var | Valeurs | Rôle |
|---|---|---|
| `PC15_PAYLOAD_FMT` | `ANS0` \| `RAW` | Choix du format de payload (ANS0 par défaut ; RAW pour debug). |
| `PC15_SCORE_BITS` | `table` \| `exact` | Estimation des bits (rapide vs encodage réel rANS). |
| `PC15_RANS_TABLES` | chemin | Répertoire custom pour les tables rANS gelées (`*.json`). |
| `PC15_GENS_JSON` | chemin | (Optionnel) Override du mapping `gen_id ↔ gen_name`. |
| `PC15_ALLOW_CPU_TESTS` | `1` | Force des chemins CPU-friendly (utile sans GPU). |

---

### Extrait d’usage

```python
import torch
from pc15codec import CodecConfig, DEFAULT_TABLE_ID
from pc15codec.codec import encode_y, decode_y

# Image Y en [-1,1] (ex. rampe synthétique)
H, W = 256, 256
y = torch.linspace(-1, 1, steps=H).view(1,1,H,1) * torch.ones(1,1,1,W)

cfg = CodecConfig(
    tile=64, overlap=16, seed=2025,
    lambda_rd=0.015, alpha_mix=0.7,
    rans_table_id=DEFAULT_TABLE_ID
)

enc = encode_y(y, cfg)          # -> {"bitstream", "bpp", "stats", ...}
yhat = decode_y(enc["bitstream"])  # -> torch.Tensor [1,1,H,W]

# Déterminisme (mêmes bytes)
enc2 = encode_y(y, cfg)
assert enc2["bitstream"] == enc["bitstream"]
```


## Codec (encode / decode)
::: pc15codec.codec
    options:
      members: true
      filters: ["!^_"]
::: pc15codec.bitstream
    options:
      members: true
      filters: ["!^_"]
::: pc15codec.tiling
    options:
      members: true
      filters: ["!^_"]
::: pc15codec.search
    options:
      members: true
      filters: ["!^_"]
::: pc15codec.rans
    options:
      members: true
      filters: ["!^_"]
::: pc15codec.payload
    options:
      members: true
      filters: ["!^_"]
::: pc15codec.symbols
    options:
      members: true
      filters: ["!^_"]
::: pc15codec.qv
    options:
      members: true
      filters: ["!^_"]
