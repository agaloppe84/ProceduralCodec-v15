from __future__ import annotations
import argparse, logging, sys, os, json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any

from .common import setup_logging, ensure_dir, list_images, RunMeta, pick_device, set_determinism, looks_like_pc15

# APIs directes pour fallback
from pc15data import to_luma_tensor
from pc15codec import encode_y

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="PC15 — Encode batch d'images en .pc15 (Y)")
    p.add_argument("--images", required=True, help="Dossier d'images")
    p.add_argument("--out", required=True, help="Dossier de sortie .pc15")
    p.add_argument("--manifest", default=None, help="(Optionnel) chemin manifeste JSON (utilisé si orchestrateur dispo)")
    p.add_argument("--config", default=None, help="(Optionnel) JSON cfg (tile/overlap/lambda/alpha)")
    p.add_argument("--tile", type=int)
    p.add_argument("--overlap", type=int)
    p.add_argument("--lambda", dest="lambda_", type=float)
    p.add_argument("--alpha", type=float)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--stats-jsonl", default=None, help="(Optionnel) JSONL stats par image (si orchestrateur, délégué)")
    p.add_argument("--resume", action="store_true", help="Skip si sortie existe et valide")
    p.add_argument("--only-plan", action="store_true", help="(Si orchestrateur) créer le manifeste et s'arrêter")
    p.add_argument("--force-cpu", action="store_true")
    p.add_argument("--log-file", default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)

def _merge_cfg(args) -> Dict[str, Any]:
    cfg = {}
    if args.config:
        try:
            cfg.update(json.loads(Path(args.config).read_text(encoding="utf-8")))
        except Exception as e:
            logging.warning("Config illisible (%s): %s", args.config, e)
    if args.tile is not None: cfg["tile"] = args.tile
    if args.overlap is not None: cfg["overlap"] = args.overlap
    if args.lambda_ is not None: cfg["lambda"] = args.lambda_
    if args.alpha is not None: cfg["alpha"] = args.alpha
    cfg.setdefault("tile", 256)
    cfg.setdefault("overlap", 24)
    cfg.setdefault("lambda", 0.015)
    cfg.setdefault("alpha", 0.7)
    return cfg

def _try_orchestrator(args, cfg: Dict[str, Any]) -> bool:
    # Si l'orchestrateur est installé, déléguer : plan → run
    try:
        from pc15wf.orchestrator import plan_encode, run_encode  # type: ignore
    except Exception:
        return False

    mani_path = args.manifest or None
    if mani_path is None or not Path(mani_path).exists():
        mani_path = plan_encode(args.images, args.out, cfg, manifest_path=args.manifest)

    # ✅ corrige le bug: --only-plan => args.only_plan
    if getattr(args, "only_plan", False):
        logging.info("Manifeste créé: %s", mani_path)
        return True

    summary = run_encode(mani_path, resume=args.resume, stats_jsonl=args.stats_jsonl)
    logging.info("Terminé: %d/%d OK, %d erreurs", summary["done"], summary["total"], summary["errors"])
    return True


def _fallback_direct(args, cfg: Dict[str, Any]) -> int:
    device = pick_device(force_cpu=args.force_cpu)
    set_determinism(args.seed)
    meta = RunMeta.collect(seed=args.seed, device=device)

    grid_cfg = {"tile": cfg["tile"], "overlap": cfg["overlap"], "lambda": cfg["lambda"], "alpha": cfg["alpha"]}
    src = Path(args.images)
    out_dir = Path(args.out); ensure_dir(out_dir)
    imgs = list_images(src)
    if not imgs:
        logging.error("Aucune image trouvée dans %s", src); return 2

    # manifest facultatif pour tracer le run
    manifest = {
        "run": asdict(meta),
        "cfg": grid_cfg,
        "inputs": [str(p) for p in imgs],
        "outputs": [],
    }

    ok = 0
    for i, path in enumerate(imgs, 1):
        out_bitstream = out_dir / f"{path.stem}.pc15"
        if args.resume and out_bitstream.exists() and looks_like_pc15(out_bitstream):
            logging.info("[%d/%d] skip: %s", i, len(imgs), out_bitstream)
            manifest["outputs"].append(str(out_bitstream)); ok += 1; continue

        try:
            logging.info("[%d/%d] encode: %s", i, len(imgs), path)
            img_y = to_luma_tensor(path)                               # [1,1,H,W] in [-1,1]
            enc = encode_y(img_y, cfg=grid_cfg)                        # {bitstream: bytes, bpp: float, stats: ...}
            tmp = out_bitstream.with_suffix(out_bitstream.suffix + ".tmp")
            with open(tmp, "wb") as f: f.write(enc["bitstream"])
            os.replace(tmp, out_bitstream)
            manifest["outputs"].append(str(out_bitstream))
            if args.stats_jsonl:
                Path(args.stats_jsonl).parent.mkdir(parents=True, exist_ok=True)
                with open(args.stats_jsonl, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"event":"encode_done","path":str(out_bitstream),"bpp":enc.get("bpp",-1.0)}, ensure_ascii=False)+"\n")
            logging.info("→ OK bpp=%.3f → %s", enc.get("bpp", -1.0), out_bitstream)
            ok += 1
        except Exception as e:
            logging.exception("Échec encodage %s: %s", path, e)

    if args.manifest:
        Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
        Path(args.manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    logging.info("Terminé: %d/%d encodées", ok, len(imgs))
    return 0 if ok == len(imgs) else 1

def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging(Path(args.log_file) if args.log_file else None, verbose=args.verbose)
    cfg = _merge_cfg(args)

    # tente orchestrateur d'abord (si présent), sinon fallback direct.
    if _try_orchestrator(args, cfg):
        return 0
    return _fallback_direct(args, cfg)

if __name__ == "__main__":
    sys.exit(main())
