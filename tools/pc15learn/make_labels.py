
from __future__ import annotations
import argparse, json, sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Dict, Any, Iterable, Tuple

import numpy as np
from PIL import Image

from .paths import PathsConfig

@dataclass
class TileLabel:
    image_path: str
    rel_path: str
    tile_id: int
    row: int
    col: int
    y_mean: float
    y_std: float
    grad_mean: float
    label: str
    complexity: float
    tile_png: str

def _to_luma_u8(p: Path) -> np.ndarray:
    im = Image.open(p).convert("L")
    return np.asarray(im, dtype=np.uint8)

def _tile(u8: np.ndarray, tile: int, overlap: int) -> Iterable[Tuple[int,int,np.ndarray]]:
    H, W = u8.shape[:2]
    stride = max(1, tile - overlap)
    tid = 0
    for r in range(0, max(1, H - tile + 1), stride):
        for c in range(0, max(1, W - tile + 1), stride):
            patch = u8[r:r+tile, c:c+tile]
            if patch.shape[0] != tile or patch.shape[1] != tile:
                pad_r = tile - patch.shape[0]
                pad_c = tile - patch.shape[1]
                patch = np.pad(patch, ((0,pad_r),(0,pad_c)), mode="edge")
            yield (tid, r, c, patch)
            tid += 1

def _features_and_label(patch_u8: np.ndarray) -> Dict[str, Any]:
    y = (patch_u8.astype(np.float32)/127.5 - 1.0)
    gy, gx = np.gradient(y)
    grad = np.sqrt(gx*gx + gy*gy)
    grad_m = float(grad.mean())
    mu = float(y.mean()); sd = float(y.std())
    if grad_m < 0.08 and sd < 0.15: lab = "smooth"
    elif grad_m > 0.20 or sd > 0.30: lab = "edgy"
    else: lab = "mid"
    comp = float(np.clip((grad_m - 0.05) / 0.30, 0.0, 1.0))
    return {"y_mean": mu, "y_std": sd, "grad_mean": grad_m, "label": lab, "complexity": comp}

def discover_images(folder: Path) -> List[Path]:
    exts = {".png",".jpg",".jpeg",".bmp",".webp",".tif",".tiff"}
    out = []
    for p in folder.rglob("*"):
        if p.suffix.lower() in exts and p.is_file():
            out.append(p)
    return sorted(out)

def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser("pc15learn.make_labels — build tiles JSONL with heuristic labels")
    ap.add_argument("--images", required=True, help="Folder with images (can be on Drive)")
    ap.add_argument("--out", default="datasets/tiles.jsonl", help="Output JSONL path")
    ap.add_argument("--tiles-out", default="datasets/tiles_png", help="Where to store tile PNGs")
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=24)
    ap.add_argument("--max-images", type=int, default=0, help="0 = no limit")
    ap.add_argument("--rel-root", default="", help="Prefix to strip from image paths to compute rel_path")
    args = ap.parse_args(argv)

    pcfg = PathsConfig()
    images_root = Path(args.images).expanduser().resolve()
    out_jsonl = Path(args.out).expanduser().resolve()
    out_tiles = Path(args.tiles_out).expanduser().resolve()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_tiles.mkdir(parents=True, exist_ok=True)

    paths = discover_images(images_root)
    if args.max_images > 0:
        paths = paths[:args.max_images]
    rel_root = Path(args.rel_root).expanduser().resolve() if args.rel_root else images_root

    n_written = 0
    with out_jsonl.open("w", encoding="utf-8") as f:
        for img_p in paths:
            try:
                u8 = _to_luma_u8(img_p)
            except Exception as e:
                print(f"[WARN] skip {img_p}: {e}", file=sys.stderr)
                continue
            for tid, r, c, patch in _tile(u8, args.tile, args.overlap):
                feats = _features_and_label(patch)
                tile_name = f"{img_p.stem}_r{r}_c{c}.png"
                tile_path = out_tiles / tile_name
                Image.fromarray(patch, mode="L").save(tile_path)
                rec = TileLabel(
                    image_path=str(img_p),
                    rel_path=str(img_p.resolve().relative_to(rel_root)),
                    tile_id=int(tid), row=int(r), col=int(c),
                    tile_png=str(tile_path),
                    **feats
                )
                f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
                n_written += 1
    print(json.dumps({
        "status": "ok", "written": n_written,
        "out_jsonl": str(out_jsonl), "tiles_dir": str(out_tiles),
        "images_root": str(images_root), "paths_config": pcfg.to_dict(),
    }, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
