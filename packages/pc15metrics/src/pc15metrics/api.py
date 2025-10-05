from __future__ import annotations
from typing import Protocol
import torch

class Metric(Protocol):
    def __call__(self, y: torch.Tensor, yhat: torch.Tensor) -> torch.Tensor: ...
