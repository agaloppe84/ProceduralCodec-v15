import torch
from pc15codec.search import SearchCfg, score_batch_bits

def test_score_batch_bits_shapes_and_monotonicity():
    cfg = SearchCfg(lambda_rd=0.02, metric="ssim", alpha=0.7)
    y = torch.zeros((1,1,16,16), dtype=torch.float32)
    # Trois candidats: parfait, bruit léger, bruit fort
    synth = torch.stack([y[0], y[0] + 0.05, y[0] + 0.3], dim=0)
    bits = [10.0, 20.0, 30.0]
    out = score_batch_bits(y, synth, cfg, bits_est=bits)
    assert out.D.shape == (3,) and out.R.shape == (3,) and out.RD.shape == (3,)
    # tendance: plus de distorsion et plus de bits ⇒ score RD plus grand
    assert float(out.RD[0]) <= float(out.RD[1]) <= float(out.RD[2])
