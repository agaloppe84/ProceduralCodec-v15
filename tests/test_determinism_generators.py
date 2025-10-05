import torch
from pc15proc.register_all import register_all
from pc15proc.registry import list_generators, get
from pc15proc.params import ParamCodec

def dev():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def mid_params(info):
    out = {}
    for p in info.param_specs:
        if p.type in ("float", "int") and p.range:
            lo, hi = p.range
            x = (float(lo)+float(hi))/2.0
            out[p.name] = int(round(x)) if p.type=="int" else x
        elif p.type == "enum":
            out[p.name] = p.enum[0]
        elif p.type == "bool":
            out[p.name] = False
        else:
            out[p.name] = 0.0
    return out

def test_determinism_subset():
    register_all()
    device = dev()
    dtype = torch.float16 if device.type=='cuda' else torch.float32
    H,W = 64,64
    seeds = torch.tensor([42], device=device, dtype=torch.int64)

    for gi in list_generators():
        g = get(gi.name)
        pc = ParamCodec(g.info)
        P = pc.to_tensor(mid_params(g.info), device=device, dtype=dtype).unsqueeze(0)
        y1 = g.render((H,W), P, seeds, device=device, dtype=dtype)
        y2 = g.render((H,W), P, seeds, device=device, dtype=dtype)
        assert torch.allclose(y1, y2, atol=0, rtol=0), f"Non-deterministic: {gi.name}"
