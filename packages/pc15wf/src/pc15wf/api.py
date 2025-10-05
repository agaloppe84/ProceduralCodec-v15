from __future__ import annotations
import os, time
from pathlib import Path

def atomic_write(path: Path | str, data: bytes) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def pc15_name(dataset: str, img: str, codec: str, q: float, bpp: float, seed: int) -> str:
    return f"{dataset}_{img}_{codec}__v15__q{q}__{bpp:.3f}__s{seed}.pc15"

def log_append(path: Path | str, msg: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")

def detect_gpu(strict: bool = True) -> dict:
    from pc15core.device import cuda_info
    return cuda_info()
