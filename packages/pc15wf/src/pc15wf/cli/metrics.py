from __future__ import annotations
import argparse, csv, logging, sys
from pathlib import Path
from PIL import Image
import numpy as np

from .common import setup_logging
from pc15metrics import psnr, ssim

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="PC15 — Metrics PSNR/SSIM & RD plot")
    p.add_argument("--pairs", nargs="+", metavar=("REF:HAT"), help="Paires ref:recon (ex: ref.png:recon.png)")
    p.add_argument("--out-csv", required=True)
    p.add_argument("--rd-png", default=None, help="Optionnel: figure RD")
    p.add_argument("--log-file", default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)

def load_u8(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("L"), dtype=np.uint8)

def to_tensor_like(u8: np.ndarray):
    arr = (u8.astype(np.float32) / 127.5) - 1.0
    try:
        import torch
        return torch.from_numpy(arr)[None, None, ...]
    except Exception:
        return arr[None, None, ...]

def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging(Path(args.log_file) if args.log_file else None, verbose=args.verbose)

    rows = [("ref","hat","psnr","ssim")]
    for spec in args.pairs or []:
        try:
            ref_s, hat_s = spec.split(":", 1)
            ref, hat = Path(ref_s), Path(hat_s)
            y, yhat = to_tensor_like(load_u8(ref)), to_tensor_like(load_u8(hat))
            rows.append((str(ref), str(hat), f"{psnr(y,yhat):.3f}", f"{ssim(y,yhat):.5f}"))
        except Exception as e:
            logging.exception("Pair invalide '%s': %s", spec, e)

    out_csv = Path(args.out_csv); out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    logging.info("→ écrit %s", out_csv)

    if args.rd_png:
        try:
            from pc15viz import plot_rd
            plot_rd(out_csv, args.rd_png)
            logging.info("→ RD figure %s", args.rd_png)
        except Exception as e:
            logging.warning("plot_rd indisponible: %s", e)
    return 0

if __name__ == "__main__":
    sys.exit(main())
