import torch
from pc15metrics.psnr_ssim import psnr, ssim

def test_metrics_basic():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y = torch.zeros((2,1,32,32), device=dev)
    yhat = torch.zeros_like(y)
    _ = psnr(y, yhat)
    assert torch.all(ssim(y,yhat) >= 0.99)
