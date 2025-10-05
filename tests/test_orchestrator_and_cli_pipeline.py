import sys, types, json
from pathlib import Path
from PIL import Image
import numpy as np

# --- Mocks ------------------------------------------------------------------
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
        return {"bitstream": b"PC15" + b"\x00"*12, "bpp": 0.123}
    def decode_y(b):
        # return gray 8x8 tensor-like
        g = np.zeros((1,1,8,8), dtype=np.float32)
        return g
    ns.encode_y = encode_y
    ns.decode_y = decode_y
    return ns

def make_mock_pc15metrics():
    ns = types.SimpleNamespace()
    ns.psnr = lambda y,yhat: 40.0
    ns.ssim = lambda y,yhat: 0.99000
    return ns

# --- Test -------------------------------------------------------------------
def test_orchestrator_plan_run_and_cli(tmp_path, monkeypatch):
    # Mock external deps BEFORE importing modules under test
    monkeypatch.setitem(sys.modules, "pc15data", make_mock_pc15data())
    monkeypatch.setitem(sys.modules, "pc15codec", make_mock_pc15codec())
    monkeypatch.setitem(sys.modules, "pc15metrics", make_mock_pc15metrics())

    # Create sample inputs
    imgs_dir = tmp_path / "imgs"; imgs_dir.mkdir()
    for i in range(2):
        Image.new("RGB", (8,8), color=(i*10, i*10, i*10)).save(imgs_dir / f"{i:04d}.png")
    out_dir = tmp_path / "outputs"
    artifacts = tmp_path / "artifacts"; artifacts.mkdir()

    # Orchestrator API
    from pc15wf.orchestrator import plan_encode, run_encode
    mani_path = plan_encode(str(imgs_dir), str(out_dir), cfg={"tile":64,"overlap":8,"lambda":0.02,"alpha":0.6})
    summary = run_encode(mani_path, resume=True, stats_jsonl=str(artifacts / "run.jsonl"))
    assert summary["total"] == 2 and summary["done"] == 2 and summary["errors"] == 0

    # Verify outputs
    for p in out_dir.glob("*.pc15"):
        assert p.read_bytes().startswith(b"PC15")

    # CLI encode (should delegate to orchestrator)
    from pc15wf.cli.encode_batch import main as encode_main
    rc = encode_main(["--images", str(imgs_dir), "--out", str(out_dir), "--resume", "--stats-jsonl", str(artifacts/"cli.jsonl")])
    assert rc == 0

    # CLI decode
    from pc15wf.cli.decode import main as decode_main
    any_bs = next(out_dir.glob("*.pc15"))
    dec_out = tmp_path / "recon"
    rc = decode_main([str(any_bs), "--out", str(dec_out)])
    assert rc == 0
    assert any(dec_out.glob("*_recon.png"))

    # CLI metrics
    ref = next(imgs_dir.glob("*.png"))
    hat = next(dec_out.glob("*_recon.png"))
    from pc15wf.cli.metrics import main as metrics_main
    rc = metrics_main(["--pairs", f"{ref}:{hat}", "--out-csv", str(artifacts / "rd.csv")])
    assert rc == 0
    assert (artifacts / "rd.csv").exists()
