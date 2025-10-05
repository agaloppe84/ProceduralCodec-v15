# tests/test_generators_timing.py
import time, torch, os
from pc15proc.register_all import register_all
from pc15proc.registry import list_generators, get
from pc15proc.params import ParamCodec

ALLOW_CPU = os.getenv("PC15_ALLOW_CPU_TESTS","0")=="1"

def dev():
    if torch.cuda.is_available(): return torch.device("cuda")
    assert ALLOW_CPU, "Set PC15_ALLOW_CPU_TESTS=1"
    return torch.device("cpu")

def mid_params(info):
    out={}
    for p in info.param_specs:
        if p.type in ("float","int") and p.range:
            lo, hi = p.range; x=(float(lo)+float(hi))/2.0
            if p.type=="int": x=int(round(x))
            out[p.name]=x
        elif p.type=="enum" and p.enum: out[p.name]=p.enum[0]
        elif p.type=="bool": out[p.name]=False
        else: out[p.name]=0.0
    return out

def test_render_timing_smoke():
    register_all()
    infos = sorted(list_generators(), key=lambda i: i.name)
    device = dev()
    dtype = torch.float16 if device.type=="cuda" else torch.float32
    for gi in infos:
        g = get(gi.name)
        pc = ParamCodec(g.info)
        P = pc.to_tensor(mid_params(g.info), device=device, dtype=dtype).unsqueeze(0)
        seeds = torch.tensor([0], device=device, dtype=torch.int64)
        t0 = time.perf_counter()
        y = g.render((64,64), P, seeds, device=device, dtype=dtype)
        dt = (time.perf_counter()-t0)*1000
        print(f"[TIMING] {gi.name:<24} {dt:6.2f} ms  shape={tuple(y.shape)}")
