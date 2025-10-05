import sys, types
from pathlib import Path

def test_cli_audit_generators_produces_artifacts(tmp_path, monkeypatch):
    # Mock pc15proc.list_generators to ensure consistent output
    fake_mod = types.SimpleNamespace()
    fake_mod.list_generators = lambda: ["checkerboard", "perlin", "worley"]
    monkeypatch.setitem(sys.modules, "pc15proc", fake_mod)

    from pc15wf.cli.audit_generators import main as audit_main
    out_dir = tmp_path / "artifacts"; out_dir.mkdir()
    rc = audit_main(["--out", str(out_dir)])
    assert rc == 0
    assert (out_dir / "audit_generators.csv").exists()
    assert (out_dir / "audit_generators.json").exists()
