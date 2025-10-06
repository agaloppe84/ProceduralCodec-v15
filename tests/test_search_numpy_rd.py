import numpy as np
from pc15codec.search import score_rd_numpy

def test_score_rd_numpy_prefers_nearby_recon_and_is_finite():
    rng = np.random.default_rng(0)
    y = rng.uniform(-1.0, 1.0, size=(16,16)).astype(np.float32)
    y_good = y + rng.normal(0, 0.01, size=(16,16)).astype(np.float32)
    y_bad  = rng.uniform(-1.0, 1.0, size=(16,16)).astype(np.float32)
    s_good = score_rd_numpy(y, y_good, lam=0.02, alpha=0.7, bits_est=100.0, metric="ssim")
    s_bad  = score_rd_numpy(y, y_bad,  lam=0.02, alpha=0.7, bits_est=100.0, metric="ssim")
    assert np.isfinite(s_good) and np.isfinite(s_bad)
    assert s_good < s_bad
