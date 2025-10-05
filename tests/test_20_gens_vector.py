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

def test_register_and_render_smoke():
    register_all()
    infos = list_generators()
    assert len(infos) >= 10
    device = dev()
    dtype = torch.float16 if device.type=="cuda" else torch.float32

    names = ["STRIPES","VALUE_NOISE","PERLIN","WORLEY","BRICK","SUNBURST"]
    for name in names:
        g = get(name)
        pc = ParamCodec(g.info)
        params = {}
        for p in g.info.param_specs:
            if p.type in ("float","int") and p.range is not None:
                lo, hi = p.range
                x = (lo + hi)/2.0
                if p.type=="int":
                    x = int(x)
                params[p.name] = x
            elif p.type=="enum" and p.enum is not None:
                params[p.name] = p.enum[0]
            elif p.type=="bool":
                params[p.name] = False
            else:
                params[p.name] = 0.0
        P = pc.to_tensor(params, device=device, dtype=dtype).unsqueeze(0)
        seeds = torch.tensor([0], device=device, dtype=torch.int64)
        y = g.render((128,128), P, seeds, device=device, dtype=dtype)
        assert y.shape == (1,1,128,128)
