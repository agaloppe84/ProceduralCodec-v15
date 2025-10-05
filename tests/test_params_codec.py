from pc15proc.params import ParamCodec
from pc15proc.stripes import GEN

def test_paramcodec_grid_and_roundtrip():
    pc = ParamCodec(GEN.info)
    grid = pc.grid()
    assert grid.ndim == 2
    vec = pc.to_tensor({"freq": 6.0, "angle_deg": 0.0, "phase": 0.0}, device=None, dtype=None)
    d = pc.from_tensor(vec)
    assert set(d.keys()) == {"freq", "angle_deg", "phase"}
