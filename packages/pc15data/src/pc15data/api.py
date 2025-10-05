from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image

@dataclass
class ImageItem:
    id: str
    path: str
    checksum: str | None = None

@dataclass
class Manifest:
    dataset: str
    images: list[ImageItem]

def ensure_symlink(src: str | Path, dst: str | Path) -> None:
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists() or dst.is_symlink():
            try:
                if dst.resolve() == src.resolve():
                    return
            except Exception:
                pass
            if dst.is_dir() and not dst.is_symlink():
                raise RuntimeError(f"{dst} existe et n'est pas un symlink")
            dst.unlink(missing_ok=True)
        os_symlink = getattr(__import__("os"), "symlink")
        os_symlink(src, dst)
    except Exception as e:
        raise RuntimeError(f"ensure_symlink failed: {e}") from e

def scan_images(root: str | Path) -> list[Path]:
    root = Path(root)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]

def to_luma_tensor(path: str | Path, *, device=None, dtype=None):
    try:
        import torch
    except Exception as e:
        from pc15core.errors import MissingCudaError
        raise MissingCudaError(f"Torch absent: {e}") from e
    img = Image.open(path).convert("YCbCr")
    y, _, _ = img.split()
    y_np = np.array(y, dtype=np.float32)
    y_np = y_np / 127.5 - 1.0
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        dtype = torch.float16 if device.type == "cuda" else torch.float32
    t = torch.from_numpy(y_np).to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
    return t
