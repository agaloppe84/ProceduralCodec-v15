
from __future__ import annotations
import os, json, time, logging, hashlib, traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List

# --- Local helpers (no CLI dependency) --------------------------------------

MAGIC = b"PC15"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def _write_json_atomic(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)

def _read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))

def _looks_like_pc15(p: Path) -> bool:
    try:
        with open(p, "rb") as f:
            head = f.read(4)
        return head == MAGIC and p.stat().st_size > 8
    except Exception:
        return False

def _sha256(p: Path, nbytes: int = 1_048_576) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(nbytes)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def _list_images(root: Path, recursive: bool = True) -> list[Path]:
    it = root.rglob("*") if recursive else root.glob("*")
    return [p for p in it if p.suffix.lower() in IMG_EXTS]

# optional dependency on pc15wf.api (atomic_write)
def _atomic_write(path: Path, data: bytes) -> None:
    try:
        from pc15wf.api import atomic_write as _aw  # type: ignore
        _aw(path, data)
    except Exception:
        # fallback simple (sans fsync)
        path = Path(path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)

@dataclass
class EncodeCfg:
    tile: int = 256
    overlap: int = 24
    lambda_: float = 0.015
    alpha: float = 0.7
    fp16: bool = True
    channels_last: bool = True

    @staticmethod
    def from_sources(cfg: Dict[str, Any] | None, read_env: bool = True) -> "EncodeCfg":
        base = dict(tile=256, overlap=24, lambda_=0.015, alpha=0.7, fp16=True, channels_last=True)
        if cfg:
            # support key 'lambda' from external cfg
            patched = {("lambda_" if k == "lambda" else k): v for k, v in cfg.items()}
            base.update(patched)
        if read_env:
            def _envf(name, cast, default):
                v = os.getenv(name)
                return cast(v) if v is not None else default
            base["tile"] = _envf("PC15_TILE", int, base["tile"])
            base["overlap"] = _envf("PC15_OVERLAP", int, base["overlap"])
            base["lambda_"] = _envf("PC15_LAMBDA", float, base["lambda_"])
            base["alpha"] = _envf("PC15_ALPHA", float, base["alpha"])
            base["fp16"] = bool(int(os.getenv("PC15_FP16", "1" if base["fp16"] else "0")))
            base["channels_last"] = bool(int(os.getenv("PC15_CHANNELS_LAST", "1" if base["channels_last"] else "0")))
        return EncodeCfg(**base)

    def to_codec_cfg(self) -> Dict[str, Any]:
        # map to encode_y expected cfg
        return {"tile": self.tile, "overlap": self.overlap, "lambda": self.lambda_, "alpha": self.alpha}

# --- Public Orchestration API ----------------------------------------------

def plan_encode(images_dir: str, out_dir: str, cfg: Dict[str, Any] | None = None,
                manifest_path: Optional[str] = None, recursive: bool = True) -> str:
    """Scan images_dir and produce a manifest to encode them into out_dir."""
    src = Path(images_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    images = _list_images(src, recursive=recursive)
    if not images:
        raise RuntimeError(f"Aucune image trouvée sous: {src}")

    cfg_obj = EncodeCfg.from_sources(cfg or {})
    items: list[dict[str, Any]] = []
    for p in images:
        stem = p.stem
        items.append({
            "id": stem,
            "in": str(p),
            "out": str(out / f"{stem}.pc15"),
            "state": "todo",
        })

    mani = {
        "kind": "pc15_encode_manifest_v1",
        "run": {"start_ts": time.time()},
        "cfg": cfg_obj.__dict__,  # keep full cfg
        "items": items,
    }

    if manifest_path is None:
        mdir = out.parent / "manifests"
        mdir.mkdir(parents=True, exist_ok=True)
        manifest_path = str(mdir / f"encode_{int(time.time())}.json")
    mp = Path(manifest_path)
    _write_json_atomic(mp, mani)
    return str(mp)

def run_encode(manifest_path: str, resume: bool = True, stats_jsonl: Optional[str] = None) -> Dict[str, Any]:
    """Execute an encoding manifest (single-process with per-item file locks)."""
    # Lazy imports to avoid hard dependencies on import
    from pc15data import to_luma_tensor  # type: ignore
    from pc15codec import encode_y       # type: ignore

    mp = Path(manifest_path)
    if not mp.exists():
        raise FileNotFoundError(f"Manifest introuvable: {mp}")
    mani = _read_json(mp)
    cfg = EncodeCfg.from_sources(mani.get("cfg", {}))
    codec_cfg = cfg.to_codec_cfg()

    stats_fp: Optional[Path] = Path(stats_jsonl) if stats_jsonl else None
    if stats_fp:
        stats_fp.parent.mkdir(parents=True, exist_ok=True)

    total = len(mani["items"]); done = 0; errors = 0
    for it in mani["items"]:
        out_p = Path(it["out"])
        # fast resume
        if resume and out_p.exists() and _looks_like_pc15(out_p):
            it["state"] = "done"
            it["size"] = out_p.stat().st_size
            it["sha256"] = _sha256(out_p)
            done += 1
            continue

        lock = out_p.with_suffix(out_p.suffix + ".lock")
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except FileExistsError:
            # another process might be doing it → skip here
            continue

        try:
            it["state"] = "running"
            it["t0"] = time.time()
            y = to_luma_tensor(it["in"])  # [1,1,H,W] in [-1,1]
            enc = encode_y(y, cfg=codec_cfg)
            _atomic_write(out_p, enc["bitstream"])
            it["state"] = "done"
            it["bpp"] = enc.get("bpp", -1.0)
            it["elapsed_s"] = time.time() - it["t0"]
            it["size"] = out_p.stat().st_size
            it["sha256"] = _sha256(out_p)
            done += 1
            if stats_fp:
                with open(stats_fp, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "event": "encode_done",
                        "id": it["id"],
                        "in": it["in"],
                        "out": it["out"],
                        "bpp": it.get("bpp", -1.0),
                        "elapsed_s": it["elapsed_s"],
                    }, ensure_ascii=False) + "\n")
        except Exception as e:
            it["state"] = "error"
            it["error"] = f"{type(e).__name__}: {e}"
            it["traceback"] = traceback.format_exc(limit=3)
            errors += 1
        finally:
            try: os.remove(lock)
            except Exception: pass
            _write_json_atomic(mp, mani)  # checkpoint after each item

    mani.setdefault("run", {})["end_ts"] = time.time()
    _write_json_atomic(mp, mani)
    return {"total": total, "done": done, "errors": errors}
