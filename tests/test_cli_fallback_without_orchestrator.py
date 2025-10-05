import sys, types, builtins
from pathlib import Path
from PIL import Image
import numpy as np

def make_mock_pc15data():
    ns = types.SimpleNamespace()
    def to_luma_tensor(p):
        arr = (np.array(Image.open(p).convert("L"), dtype=np.float32)/127.5)-1.0
        return arr[None,None,...]
    ns.to_luma_tensor = to_luma_tensor
    return ns

def make_mock_pc15codec():
    ns = types.SimpleNamespace()
    def encode_y(y, cfg):
        return {"bitstream": b"PC15" + b"\x01"*8, "bpp": 0.111}
    ns.encode_y = encode_y
    return ns

def test_cli_encode_fallback(tmp_path, monkeypatch):
    # Force ImportError on orchestrator to test fallback
    real_import = builtins.__import__
    def fake_import(name, *a, **kw):
        if name == "pc15wf.orchestrator":
            raise ImportError("forced for test")
        return real_import(name, *a, **kw)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    # Mock deps
    monkeypatch.setitem(sys.modules, "pc15data", make_mock_pc15data())
    monkeypatch.setitem(sys.modules, "pc15codec", make_mock_pc15codec())

    # Create input
    imgs_dir = tmp_path / "imgs"; imgs_dir.mkdir()
    Image.new("RGB", (8,8), color=(128,128,128)).save(imgs_dir / "a.png")
    out_dir = tmp_path / "outputs"

    # Run CLI
    from pc15wf.cli.encode_batch import main as encode_main
    rc = encode_main(["--images", str(imgs_dir), "--out", str(out_dir), "--resume", "--seed", "1234"])
    assert rc == 0
    assert any(out_dir.glob("*.pc15"))
