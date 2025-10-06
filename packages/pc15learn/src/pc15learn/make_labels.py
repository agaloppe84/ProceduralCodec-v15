from __future__ import annotations
import argparse
from pathlib import Path

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("")  # JSONL vide (stub)
    print(f"✓ wrote labels stub → {out}")

if __name__ == "__main__":
    main()
