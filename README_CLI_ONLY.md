# PC15 — CLI (orchestrator-ready), v15

Ce bundle fournit un **CLI robuste** qui **délègue automatiquement** à `pc15wf.orchestrator`
s'il est présent, sinon **retombe** sur un chemin direct API (`pc15data` + `pc15codec`).

## Fichiers à placer
Copier le contenu du dossier `packages/pc15wf/src/pc15wf/cli/` dans votre repo au même chemin.

## Entry points à ajouter dans `packages/pc15wf/pyproject.toml`
```toml
[project.scripts]
pc15-encode  = "pc15wf.cli.encode_batch:main"
pc15-decode  = "pc15wf.cli.decode:main"
pc15-metrics = "pc15wf.cli.metrics:main"
pc15-audit   = "pc15wf.cli.audit_generators:main"
```

## Usage
```bash
pip install -e packages/pc15wf

pc15-audit --out artifacts

pc15-encode --images /path/to/images --out outputs \
            --tile 256 --overlap 24 --lambda 0.015 --alpha 0.7 \
            --resume --seed 1234 --stats-jsonl artifacts/run.jsonl

pc15-decode outputs/0001.pc15 --out outputs

pc15-metrics --pairs /path/to/images/0001.png:outputs/0001_recon.png \
             --out-csv artifacts/rd.csv --rd-png artifacts/rd.png
```

## Notes
- `pc15-encode` : si `pc15wf.orchestrator` existe, utilise `plan_encode/run_encode`.
  Sinon encode directement en boucle en appelant `pc15codec.encode_y`.
- `--config codec.json` permet de charger une config et de la surcharger par flags.
- `--stats-jsonl` fonctionne dans les deux modes (orchestrateur gère lui-même sinon fallback simple).
- Validation `.pc15` minimale (magic `PC15`) pour `--resume` ; remplacez par header+CRC quand la spec est figée.
