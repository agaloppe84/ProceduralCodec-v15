from __future__ import annotations
import argparse, csv, json, logging, sys
from pathlib import Path

from .common import setup_logging
from pc15proc import list_generators

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="PC15 — Audit des générateurs procéduraux")
    p.add_argument("--out", required=True, help="Dossier de sortie (csv/json)")
    p.add_argument("--log-file", default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)

def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging(Path(args.log_file) if args.log_file else None, verbose=args.verbose)

    gens = list_generators()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "audit_generators.json").write_text(
        json.dumps({"generators": gens}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with open(out_dir / "audit_generators.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["name"]); [w.writerow([g]) for g in gens]

    logging.info("Générateurs: %d", len(gens))
    return 0

if __name__ == "__main__":
    sys.exit(main())
