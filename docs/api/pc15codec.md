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

## 7) Critères d’acceptation (Step 3)
- `write_stream_v15/read_stream_v15` **inchangés** (déjà OK). :contentReference[oaicite:10]{index=10}
- Les `.pc15` produits contiennent des `TileRec.payload_fmt == 0 (ANS0)` et se **décodent sans erreur** (round-trip symbolique).
- `encode_tile_payload` / `decode_tile_payload` **CPU-only** OK.
- Tables gelées chargées via `load_table_by_id("v15_default")`. :contentReference[oaicite:11]{index=11}
- Back-compat : `PC15_PAYLOAD_FMT=RAW` → chemin actuel **RAW** inchangé. :contentReference[oaicite:12]{index=12}

---

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
