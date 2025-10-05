from __future__ import annotations
import argparse, json, os, sys, time, hashlib, logging, random, subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, List

try:
    import torch  # optionnel
except Exception:
    torch = None  # type: ignore

def setup_logging(log_file: Optional[Path], verbose: bool = True) -> None:
    log_fmt = "[%(asctime)s] %(levelname)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=level, format=log_fmt, datefmt=datefmt, handlers=handlers)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def list_images(root: Path, exts=(".png",".jpg",".jpeg",".bmp",".tif",".tiff")) -> list[Path]:
    root = Path(root)
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]

MAGIC = b"PC15"
def looks_like_pc15(p: Path) -> bool:
    try:
        with open(p, "rb") as f:
            return f.read(4) == MAGIC
    except Exception:
        return False

@dataclass
class RunMeta:
    cmd: list[str]
    start_ts: float
    git_commit: Optional[str]
    torch: Optional[str]
    cuda: Optional[str]
    device: str
    seed: Optional[int]

    @staticmethod
    def collect(seed: Optional[int] = None, device: str = "auto") -> "RunMeta":
        git_commit = None
        try:
            git_commit = subprocess.check_output(
                ["git","rev-parse","--short","HEAD"], text=True
            ).strip()
        except Exception:
            pass
        torch_v = getattr(torch, "__version__", None) if torch else None
        cuda_v = None
        if torch and getattr(torch, "cuda", None) and torch.cuda.is_available():
            try:
                cuda_v = torch.version.cuda
            except Exception:
                pass
        return RunMeta(sys.argv[:], time.time(), git_commit, torch_v, cuda_v, device, seed)

def pick_device(force_cpu: bool = False) -> str:
    if force_cpu or (torch is None):
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

def set_determinism(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    if torch:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
