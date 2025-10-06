
import torch
from pc15vq.qv import train_qv_kmeans, quantize_batch, dequantize_batch, quantize_params, dequantize_params

def test_qv_kmeans_roundtrip_small():
    torch.manual_seed(0)
    # Données 2D autour de 3 centres
    centers = torch.tensor([[0.0,0.0],[3.0,3.0],[0.0,3.0]], dtype=torch.float32)
    data = torch.cat([c + 0.1*torch.randn(50,2) for c in centers], dim=0)
    cb = train_qv_kmeans(data, K=3, iters=15, seed=123)
    # Quantif/déquantif
    idx, res = quantize_batch(data, cb)
    recon = dequantize_batch(idx, res, cb)
    # L'erreur RMS doit être faible (~ bruit)
    rmse = torch.sqrt(torch.mean((recon - data)**2)).item()
    assert rmse < 0.12

def test_qv_single_param_api_shapes():
    p = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float32)
    cb = torch.stack([p, p+1.0], dim=0)
    idx, residual = quantize_params(p, cb)
    p2 = dequantize_params(idx, residual, cb)
    assert idx in (0,1)
    assert p2.shape == p.shape
    assert torch.allclose(p2, p)
