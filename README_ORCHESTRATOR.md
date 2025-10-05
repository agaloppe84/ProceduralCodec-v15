
# PC15 Orchestrator (v15)

Ce module fournit une orchestration **légère et fiable** pour PC15 :
- `plan_encode(images_dir, out_dir, cfg, manifest_path=None)` → génère un manifeste
- `run_encode(manifest_path, resume=True, stats_jsonl=None)` → exécute l'encodage avec idempotence

## Paramètres
- CFG (`dict`) et variables d'environnement lues automatiquement :
  - `PC15_TILE`, `PC15_OVERLAP`, `PC15_LAMBDA`, `PC15_ALPHA`
  - `PC15_FP16`, `PC15_CHANNELS_LAST` (conservés dans le manifeste, passés au besoin plus tard)

## États & idempotence
- Par item: `state = todo | running | done | error`
- `--resume` / `resume=True` : skip si la sortie `.pc15` existe et a le magic `PC15`
- Checkpoints: le manifeste est ré-écrit après chaque item
- Verrouillage: `.lock` par sortie pour éviter les doublons si processus concurrents

## Intégration CLI
- Le CLI `pc15-encode` essaye d'importer `pc15wf.orchestrator` et, si présent, utilise
  `plan_encode` puis `run_encode`. Sinon il retombe sur un encodeur direct.

## Exemple d'utilisation Python
```python
from pc15wf.orchestrator import plan_encode, run_encode
m = plan_encode("imgs/", "outputs/", cfg={"tile":256,"overlap":24,"lambda":0.015,"alpha":0.7})
summary = run_encode(m, resume=True, stats_jsonl="artifacts/run.jsonl")
print(summary)
```
