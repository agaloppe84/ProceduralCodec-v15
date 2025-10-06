import torch
from pc15codec.search import SearchCfg, score_batch, make_border_mask

def test_seam_weight_increases_D_when_mask_given():
    cfg = SearchCfg(lambda_rd=0.02, metric="ssim", alpha=0.7)
    y = torch.zeros((1,1,16,16), dtype=torch.float32)
    synth = torch.stack([y[0] + 0.2], dim=0)
    base = score_batch(y, synth, cfg, seam_weight=0.0, border_mask=None)
    bm = make_border_mask(16, 16, width=8, device=y.device, dtype=y.dtype)
    withmask = score_batch(y, synth, cfg, seam_weight=0.5, border_mask=bm)
    # La pénalité couture doit augmenter D
    assert float(withmask.D) > float(base.D)

def test_auto_mask_equals_explicit_mask_when_same_width():
    cfg = SearchCfg(lambda_rd=0.02, metric="ssim", alpha=0.7)
    y = torch.zeros((1,1,16,16), dtype=torch.float32)
    synth = torch.stack([y[0] + 0.2], dim=0)
    auto = score_batch(y, synth, cfg, seam_weight=0.5, border_mask=None)
    bm = make_border_mask(16, 16, width=8, device=y.device, dtype=y.dtype)
    explicit = score_batch(y, synth, cfg, seam_weight=0.5, border_mask=bm)
    # Le masque implicite (width=8) doit matcher le masque explicite (width=8)
    assert torch.allclose(auto.D, explicit.D)
