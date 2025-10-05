import os
import csv
import json
import time
import logging
from pathlib import Path

import torch

from pc15proc.register_all import register_all
from pc15proc.registry import list_generators, get
from pc15proc.params import ParamCodec

log = logging.getLogger("pc15.audit.generators")

def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _mid_params(info):
    out = {}
    for p in info.param_specs:
        if p.type in ("float", "int") and p.range:
            lo, hi = p.range
            x = (float(lo) + float(hi)) / 2.0
            out[p.name] = int(round(x)) if p.type == "int" else x
        elif p.type == "enum":
            out[p.name] = p.enum[0]
        elif p.type == "bool":
            out[p.name] = False
        else:
            out[p.name] = 0.0
    return out

def test_generators_batch_audit():
    """
    Audit batch (shape, finiteness, range, déterminisme) + timings.
    Écrit un CSV/JSON dans ./artifacts/ et logge chaque générateur.
    Fait échouer le test si un générateur viole les invariants.
    """
    # Paramétrage via variables d'env si besoin
    H = int(os.getenv("PC15_AUDIT_H", "64"))
    W = int(os.getenv("PC15_AUDIT_W", "64"))
    B = int(os.getenv("PC15_AUDIT_B", "16"))

    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "audit_generators.csv"
    out_json = out_dir / "audit_generators.json"

    # Logs live (si pytest -o log_cli=true)
    log.info("=== Audit generators === HxW=%dx%d  B=%d", H, W, B)

    register_all()
    device = _device()
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    rows = []
    problems = []

    for gi in sorted(list_generators(), key=lambda x: x.name):
        name = gi.name
        try:
            g = get(name)
            pc = ParamCodec(g.info)
            params = _mid_params(g.info)

            # Batch params: on duplique les mêmes params pour B; on varie les seeds
            P1 = pc.to_tensor(params, device=device, dtype=dtype).unsqueeze(0)
            P = P1.expand(B, -1)
            seeds = torch.arange(B, device=device, dtype=torch.int64)

            t0 = time.perf_counter()
            y = g.render((H, W), P, seeds, device=device, dtype=dtype)
            t1 = time.perf_counter()

            # Invariants
            ok_shape = tuple(y.shape) == (B, 1, H, W)
            ok_finite = bool(torch.isfinite(y).all())
            ok_range = bool(((y >= -1) & (y <= 1)).all())

            # Déterminisme par échantillon (mêmes seeds => même sortie)
            y2 = g.render((H, W), P, seeds, device=device, dtype=dtype)
            ok_det = bool(torch.allclose(y, y2, atol=0, rtol=0))

            y32 = y.detach().to(torch.float32).cpu()
            mn = float(y32.min())
            mx = float(y32.max())
            mean = float(y32.mean())
            checksum = float(y32.sum())

            ms = (t1 - t0) * 1000.0
            per_img = ms / max(1, B)

            log.info(
                "[%-18s] %6.2f ms  per_img=%6.3f ms  shape=%s finite=%s range=%s det=%s  "
                "min=%.4f max=%.4f mean=%.4f sum=%.4f",
                name, ms, per_img, ok_shape, ok_finite, ok_range, ok_det,
                mn, mx, mean, checksum,
            )

            row = {
                "name": name,
                "time_ms_B": round(ms, 3),
                "per_img_ms": round(per_img, 4),
                "ok_shape": ok_shape,
                "ok_finite": ok_finite,
                "ok_range": ok_range,
                "ok_det": ok_det,
                "min": round(mn, 6),
                "max": round(mx, 6),
                "mean": round(mean, 6),
                "sum": round(checksum, 6),
                "H": H, "W": W, "B": B,
            }
            rows.append(row)

            if not (ok_shape and ok_finite and ok_range and ok_det):
                problems.append(row)

        except Exception as e:
            log.exception("Audit failure on %s", name)
            rows.append({"name": name, "error": repr(e), "H": H, "W": W, "B": B})
            problems.append({"name": name, "error": repr(e)})

    # CSV
    if rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            fieldnames = sorted({k for r in rows for k in r.keys()})
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    # JSON
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {"H": H, "W": W, "B": B, "device": str(device), "dtype": str(dtype), "rows": rows, "problems": problems},
            f, indent=2
        )

    log.info("Audit CSV  -> %s", out_csv)
    log.info("Audit JSON -> %s", out_json)

    # Échec si problème
    assert not problems, f"{len(problems)} générateur(s) à corriger (voir artifacts/audit_generators.*)."
