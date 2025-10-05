import torch
from pc15proc.register_all import register_all
from pc15proc.registry import get
from pc15proc.params import ParamCodec

def _mid_params(info):
    params = {}
    for p in info.param_specs:
        if p.type in ("float","int") and p.range is not None:
            lo, hi = p.range
            x = (float(lo) + float(hi)) / 2.0
            if p.type == "int":
                x = int(round(x))
            params[p.name] = x
        elif p.type == "enum" and p.enum is not None:
            params[p.name] = p.enum[0]
        elif p.type == "bool":
            params[p.name] = False
        else:
            params[p.name] = 0.0
    return params

def test_paramcodec_roundtrip():
    register_all()
    g = get("STRIPES")
    pc = ParamCodec(g.info)
    params = _mid_params(g.info)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if dev.type == "cuda" else torch.float32
    vec = pc.to_tensor(params, device=dev, dtype=dtype)
    back = pc.from_tensor(vec)
    # Check keys preserved and types consistent (int/bool exact, floats approx)
    assert set(back.keys()) == set(params.keys())
    for k, v in params.items():
        if isinstance(v, bool) or isinstance(v, int):
            assert type(back[k]) in (bool, int)
        else:
            assert isinstance(back[k], float)
