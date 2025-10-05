import re
from pathlib import Path

PAT = re.compile(r"torch\.tensor\(0x[0-9A-Fa-f]{16,}[^)]*dtype\s*=\s*torch\.int64")

def test_no_int64_hex_tensor_literals():
    root = Path("packages/pc15proc/src/pc15proc")
    offenders = []
    for p in root.rglob("*.py"):
        if "noise.py" in str(p):
            # on accepte ce fichier car on a déjà le patch via to_int64_signed
            pass
        txt = p.read_text(encoding="utf-8", errors="ignore")
        for m in PAT.finditer(txt):
            offenders.append((p, m.group(0)))
    assert not offenders, f"Replace hex int64 tensors with to_int64_signed + XOR int: {offenders}"
