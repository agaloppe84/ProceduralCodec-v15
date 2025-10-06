import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

# Import direct depuis le sous-package intégré
make_labels_main = pytest.importorskip("pc15learn.make_labels").main


def test_make_labels_tmp(tmp_path: Path):
    imgs = tmp_path / "images"
    imgs.mkdir(parents=True, exist_ok=True)

    a = (np.linspace(0, 255, 64, dtype=np.uint8)[None, :].repeat(64, axis=0))
    b = np.random.default_rng(0).integers(0, 256, size=(64, 64), dtype=np.uint8)
    Image.fromarray(a, "L").save(imgs / "grad.png")
    Image.fromarray(b, "L").save(imgs / "noise.png")

    out_jsonl = tmp_path / "tiles.jsonl"
    out_tiles = tmp_path / "tiles"

    rc = make_labels_main([
        "--images", str(imgs),
        "--out", str(out_jsonl),
        "--tiles-out", str(out_tiles),
        "--tile", "32", "--overlap", "8",
    ])

    # certains CLIs renvoient None (success), d'autres 0
    assert rc in (None, 0)
    assert out_jsonl.exists()
    lines = out_jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) > 0

    first = json.loads(lines[0])
    # garde les labels actuels si ton make_labels en émet
    assert "label" in first and first["label"] in ("smooth", "mid", "edgy")
