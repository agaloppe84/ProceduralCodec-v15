# ProceduralCodec v15 — API (patched)

API **complète et testable** pour le codec procédural par tuiles (v15).
- Packages: `pc15core`, `pc15proc`, `pc15codec`, `pc15vq`, `pc15metrics`, `pc15data`, `pc15viz`, `pc15wf`
- Tests: `pytest` (fallback CPU via `PC15_ALLOW_CPU_TESTS=1`)
- CI minimal: installe Torch CPU + lance les tests

## Dev quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r configs/requirements.dev.txt
pip install -e packages/pc15core -e packages/pc15proc -e packages/pc15codec -e packages/pc15vq -e packages/pc15metrics -e packages/pc15data -e packages/pc15viz -e packages/pc15wf

# si pas de GPU local
export PC15_ALLOW_CPU_TESTS=1
pytest -q
```
