import os, torch
from pc15proc.register_all import register_all
from pc15proc.registry import list_generators, get
from pc15proc.params import ParamCodec

ALLOW_CPU = os.getenv("PC15_ALLOW_CPU_TESTS","0")=="1"

def dev():
    if torch.cuda.is_available():
        return torch.device("cuda")
    assert ALLOW_CPU, "Set PC15_ALLOW_CPU_TESTS=1"
    return torch.device("cpu")

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

def test_autoload_and_smoke_render_subset():
    names = register_all()
    infos = list_generators()
    assert len(infos) >= 20
    device = dev()
    dtype = torch.float16 if device.type=="cuda" else torch.float32
    sample = names[:6] if len(names) >= 6 else names
    for name in sample:
        g = get(name)
        pc = ParamCodec(g.info)
        P = pc.to_tensor(_mid_params(g.info), device=device, dtype=dtype).unsqueeze(0)
        seeds = torch.tensor([0], device=device, dtype=torch.int64)
        y = g.render((64,64), P, seeds, device=device, dtype=dtype)
        assert y.shape == (1,1,64,64)
        assert torch.isfinite(y).all()
