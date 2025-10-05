import torch
from pc15proc.register_all import register_all
from pc15proc.registry import get
from pc15proc.params import ParamCodec

def dev():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_composite_smoke():
    register_all()
    g = get("COMPOSITE")
    pc = ParamCodec(g.info)
    device = dev()
    dtype = torch.float16 if device.type == 'cuda' else torch.float32

    params = {
        "base_id": 2,   # PERLIN
        "warp_id": 1,   # FBM
        "mask_id": 1,   # VORONOI_EDGES
        "pal_id":  1,   # PALETTE2
        "base_q":  0.5,
        "warp_q":  0.5,
        "mask_q":  0.5,
        "pal_q":   0.5,
    }
    P = pc.to_tensor(params, device=device, dtype=dtype).unsqueeze(0)
    seeds = torch.tensor([0], device=device, dtype=torch.int64)
    y = g.render((64, 64), P, seeds, device=device, dtype=dtype)
    assert y.shape == (1, 1, 64, 64)
    assert torch.isfinite(y).all()
    assert (y >= -1).all() and (y <= 1).all()
