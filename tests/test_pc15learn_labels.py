
import sys, json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.pc15learn.make_labels import main as make_labels_main  # noqa

def test_make_labels_tmp(tmp_path: Path):
    imgs = tmp_path / "images"
    imgs.mkdir(parents=True, exist_ok=True)
    a = (np.linspace(0,255,64, dtype=np.uint8)[None,:].repeat(64, axis=0))
    b = np.random.default_rng(0).integers(0,256, size=(64,64), dtype=np.uint8)
    Image.fromarray(a, 'L').save(imgs / "grad.png")
    Image.fromarray(b, 'L').save(imgs / "noise.png")

    out_jsonl = tmp_path / "tiles.jsonl"
    out_tiles = tmp_path / "tiles"

    rc = make_labels_main([
        "--images", str(imgs),
        "--out", str(out_jsonl),
        "--tiles-out", str(out_tiles),
        "--tile", "32", "--overlap", "8",
    ])
    assert rc == 0
    assert out_jsonl.exists()
    lines = out_jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) > 0
    first = json.loads(lines[0])
    assert "label" in first and first["label"] in ("smooth","mid","edgy")
