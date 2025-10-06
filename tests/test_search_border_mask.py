import torch
from pc15codec.search import SearchCfg, score_batch, make_border_mask

def test_score_batch_with_border_mask_changes_D():
    cfg = SearchCfg(lambda_rd=0.02, metric="ssim", alpha=0.7)
    y = torch.zeros((1,1,16,16), dtype=torch.float32)
    synth = torch.stack([y[0] + 0.2], dim=0)
    out_nomask = score_batch(y, synth, cfg, seam_weight=0.5, border_mask=None)
    bm = make_border_mask(16, 16, width=8, device=y.device, dtype=y.dtype)
    out_withmask = score_batch(y, synth, cfg, seam_weight=0.5, border_mask=bm)
    # La présence du masque borde modifie D (pas nécessairement < ou >, mais différent)
    assert float(torch.abs(out_nomask.D - out_withmask.D)) > 0.0
