from __future__ import annotations
import argparse, logging, sys
from pathlib import Path
from PIL import Image
import numpy as np

from .common import setup_logging, ensure_dir
from pc15codec import decode_y

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="PC15 — Decode .pc15 -> PNG (Y)")
    p.add_argument("bitstreams", nargs="+", help="Fichiers .pc15")
    p.add_argument("--out", required=True, help="Dossier de sortie")
    p.add_argument("--log-file", default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)

def to_u8(y_tensor) -> np.ndarray:
    try:
        import torch
        arr = y_tensor[0,0].clamp(-1,1).detach().cpu().numpy()
    except Exception:
        arr = y_tensor[0,0]
    return ((arr + 1.0) * 127.5).astype(np.uint8)

def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging(Path(args.log_file) if args.log_file else None, verbose=args.verbose)

    out_dir = Path(args.out); ensure_dir(out_dir)
    ok = 0
    for i, p in enumerate(args.bitstreams, 1):
        p = Path(p)
        try:
            logging.info("[%d/%d] decode: %s", i, len(args.bitstreams), p)
            img_y = decode_y(p.read_bytes())
            Image.fromarray(to_u8(img_y), mode="L").save(out_dir / f"{p.stem}_recon.png")
            logging.info("→ OK %s", out_dir / f"{p.stem}_recon.png")
            ok += 1
        except Exception as e:
            logging.exception("Échec decode %s: %s", p, e)
    return 0 if ok == len(args.bitstreams) else 1

if __name__ == "__main__":
    sys.exit(main())
