from __future__ import annotations
import torch
import torch.nn.functional as F

def psnr(y: torch.Tensor, yhat: torch.Tensor) -> torch.Tensor:
    mse = torch.mean((y - yhat) ** 2, dim=(1, 2, 3), keepdim=False)
    eps = 1e-12
    return 10.0 * torch.log10(1.0 / torch.clamp(mse, min=eps))

def _gauss1d(n: int = 11, sigma: float = 1.5, device=None, dtype=None):
    t = torch.arange(n, device=device, dtype=dtype) - (n - 1) / 2
    g = torch.exp(-(t**2) / (2 * sigma * sigma))
    return (g / g.sum()).view(1, 1, 1, n)

def ssim(y: torch.Tensor, yhat: torch.Tensor) -> torch.Tensor:
    B, _, H, W = y.shape
    device, dtype = y.device, y.dtype
    g = _gauss1d(device=device, dtype=dtype)
    def blur(x: torch.Tensor) -> torch.Tensor:
        x = F.conv2d(x, g, padding=(0, 5), groups=1)
        x = F.conv2d(x, g.transpose(2, 3), padding=(5, 0), groups=1)
        return x
    mu_x, mu_y = blur(y), blur(yhat)
    sigma_x = blur(y * y) - mu_x * mu_x
    sigma_y = blur(yhat * yhat) - mu_y * mu_y
    sigma_xy = blur(y * yhat) - mu_x * mu_y
    C1, C2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))
    out = ssim_map.mean(dim=(1, 2, 3)).clamp(0, 1)
    return out
