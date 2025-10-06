
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

try:
    import torch  # type: ignore
except Exception:
    torch = None

@dataclass
class HeuristicHooks:
    smooth_seed: int = 1234
    edgy_seed:   int = 987654321
    grad_thresh: float = 0.15

    def choose_seed_from_img(self, img_y) -> int:
        if torch is not None and isinstance(img_y, torch.Tensor):
            y = img_y.detach().float().cpu().numpy()
            if y.ndim == 4: y = y[0,0]
        else:
            y = np.asarray(img_y)
            if y.ndim == 3: y = y[0]
        y = y.astype(np.float32)
        m, M = float(y.min()), float(y.max())
        if M - m > 1e-8:
            y = 2.0 * (y - m) / (M - m) - 1.0
        gy, gx = np.gradient(y)
        grad_mean = float(np.mean(np.sqrt(gx*gx + gy*gy)))
        return self.edgy_seed if grad_mean > self.grad_thresh else self.smooth_seed

def encode_y_with_hooks(img_y, cfg, hooks: Optional[HeuristicHooks] = None):
    import copy
    import pc15 as pc
    cfg2 = copy.copy(cfg)
    if hooks is not None:
        seed = hooks.choose_seed_from_img(img_y)
        try:
            _ = cfg2.seed
            cfg2.seed = int(seed)
        except Exception:
            try:
                cfg2["seed"] = int(seed)
            except Exception:
                pass
    return pc.encode_y(img_y, cfg2)
